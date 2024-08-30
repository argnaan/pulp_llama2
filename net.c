#include "pmsis.h"
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "stdlib.h"
#include "stdio.h"

#include "stats.h"
#include "pulp_train.h"
#include "pulp_rmsnorm_fp16.h"
#include "conf_and_weights_fp16.h"

PI_L1 fp16 buffer_n_cores[NUM_CORES];   // for parallelized RMSNorm and softmax

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    fp16* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    fp16* rms_att_weight; // (layer, dim) rmsnorm weights
    fp16* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    fp16* wq; // (layer, dim, n_heads * head_size)
    fp16* wk; // (layer, dim, n_kv_heads * head_size)
    fp16* wv; // (layer, dim, n_kv_heads * head_size)
    fp16* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    fp16* w1; // (layer, hidden_dim, dim)
    fp16* w2; // (layer, dim, hidden_dim)
    fp16* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    fp16* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    fp16* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    fp16 *x; // activation at current time stamp (dim,)
    fp16 *xb; // same, but inside a residual branch (dim,)
    fp16 *xb2; // an additional buffer just for convenience (dim,)
    fp16 *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    fp16 *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    fp16 *q; // query (dim,)
    fp16 *k; // key (dim,)
    fp16 *v; // value (dim,)
    fp16 *att; // buffer for scores/attention values (n_heads, seq_len)
    fp16 *logits; // output logits (vocab_size, )
    // kv cache
    fp16* key_cache;   // (layer, seq_len, dim)
    fp16* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    fp16* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void memory_map_weights(TransformerWeights *w, Config* p, fp16* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(Config* config, TransformerWeights* weights, int* fd, fp16** data, ssize_t* file_size) {
    config->dim = DIM;
    config->hidden_dim = HIDDEN_DIM;
    config->n_heads = N_HEADS;
    config->n_kv_heads = N_KV_HEADS;
    config->n_layers = N_LAYERS;
    config->seq_len = SEQ_LEN;
    config->vocab_size = VOCAB_SIZE;

    int shared_weights;
    if(config->vocab_size > 0)
        shared_weights = 1;
    else{
        shared_weights = 0;
        config->vocab_size = - config->vocab_size;
    }
    fp16* weights_ptr = weights_list;
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void malloc_run_state(RunState* s, Config* p) {
    s->key_cache = KEY_CACHE;
    s->value_cache = VALUE_CACHE;
}

void build_transformer(Transformer *t) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(&t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void matmul(fp16* xout, fp16* x, fp16* w, int n, int d) {
/*
    original code: 
    int i;
    for (i = 0; i < d; i++) {
        fp16 val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
*/
    struct matMul_args_fp16 mm_args;
    mm_args.A = w;
    mm_args.B = x;
    mm_args.C = xout; 
    mm_args.N = d;
    mm_args.K = n;
    mm_args.M = 1;
    mm_args.trans_B = 0;

    pi_cl_team_fork(NUM_CORES, mv_fp16_SIMD_4x1, &mm_args);
}

struct llama2_mhsa_args_fp16{
    fp16* q;
    fp16* att;
    fp16* key_cache;
    fp16* value_cache;
    fp16* xb;
    int pos;
    int kv_dim;
    int kv_mul;
    int head_size;
    int n_heads;
    int steps;
};

void llama2_mhsa_fp16_cl(void *llama2_mhsa_args){
    struct llama2_mhsa_args_fp16* args = (struct llama2_mhsa_args_fp16*) llama2_mhsa_args;

    int pos = args->pos;
    int kv_dim = args->kv_dim;
    int kv_mul = args->kv_mul;
    int head_size = args->head_size;
    int n_heads = args->n_heads;
    const fp16 sqrt_head_size = (fp16) sqrtf(head_size);

    int id = pi_core_id();

    const uint32_t blockSize = (n_heads + NUM_CORES-1) / NUM_CORES;
    const uint32_t start = pi_core_id()*blockSize;
    const uint32_t stop = start+blockSize > n_heads ? n_heads : start+blockSize;

    for (int h = start; h < stop; h++) {
            // get the query vector for this head
            fp16* q = args->q + h * head_size;
            // attention scores for this head
            fp16* att = args->att + h * (STEPS+1);
            // iterate over all timesteps, including the current one

            fp16 max_val = -100000;
            int t;
            for(t=0; t <= pos-3; t+=4) {
                // get the key vector for this head and at this timestep
                fp16* k = args->key_cache + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                
                v2f16 temp1 = (v2f16) {0, 0};
                v2f16 temp2 = (v2f16) {0, 0};
                v2f16 temp3 = (v2f16) {0, 0};
                v2f16 temp4 = (v2f16) {0, 0};
                v2f16 A, B1, B2, B3, B4;
                for (int i = 0; i < head_size; i+=2) {
                    A = *((v2f16*) &q[i]);
                    B1 = *((v2f16*) &k[i]);
                    B2 = *((v2f16*) &k[i + kv_dim]);
                    B3 = *((v2f16*) &k[i + 2*kv_dim]);
                    B4 = *((v2f16*) &k[i + 3*kv_dim]);
                    temp1 += A * B1;
                    temp2 += A * B2;
                    temp3 += A * B3;
                    temp4 += A * B4;
                }

                // save the score to the attention buffer
                att[t] = (temp1[0] + temp1[1]) / sqrt_head_size;
                if(att[t] > max_val) 
                    max_val = att[t];
                
                att[t+1] = (temp2[0] + temp2[1]) / sqrt_head_size;
                if(att[t+1] > max_val)
                    max_val = att[t+1];
                
                att[t+2] = (temp3[0] + temp3[1]) / sqrt_head_size;
                if(att[t+2] > max_val)
                    max_val = att[t+2];
                
                att[t+3] = (temp4[0] + temp4[1]) / sqrt_head_size;
                if(att[t+3] > max_val)
                    max_val = att[t+3];
            }
            
            // leftover
            while(t <= pos) {
                // get the key vector for this head and at this timestep
                fp16* k = args->key_cache + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                
                v2f16 temp = (v2f16) {0, 0};
                v2f16 A,B;
                for (int i = 0; i < head_size; i+=2) {
                    A = *((v2f16*) &q[i]);
                    B = *((v2f16*) &k[i]);
                    temp += A * B;
                }
                // save the score to the attention buffer
                att[t] = ( temp[0] + temp[1] ) / sqrt_head_size;
                if(att[t] > max_val)
                    max_val = att[t];
                t++;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            // softmax_original_fp16(att, pos + 1);
            fp16 sum = 0.0f;
            for (int t = 0; t < pos+1; t++) {
                // FastExp
                float x = (float) (att[t] - max_val);
                x = GIST_A * x + GIST_B;
                if (x < GIST_C )    // no need to check if x > GIST_D, because x <= 0
                    x = 0.0f;

                uint32_t n = (uint32_t) (x);
                att[t] = (fp16) *(float*) &n;

                sum += att[t];
            }

            // weighted sum of the values, store back into xb
            fp16* xb = args->xb + h * head_size;
            fp16* v = args->value_cache + (h / kv_mul) * head_size;

            // for each t:  xb += v[t] * att[t];
            for(int i=0 ; i < head_size ; i+=2){            // only works with even head_size. TODO: add leftover
                v2f16 temp = (v2f16) {0, 0};
                for(int t = 0; t <= pos; t++){
                    temp += *((v2f16*)&v[i + t*kv_dim]) * (v2f16) {att[t], att[t]};
                }
                xb[i] = temp[0] / sum;
                xb[i+1] = temp[1] / sum;
            }
    }
}

struct rope_args_fp16{
    fp16* q;
    fp16* k;
    int pos;
    int dim;
    int head_size;
    int kv_dim;
};

void rope_parallelized_fp16_cl(void* void_args){

    // Works only width head_size = NUM_CORES. TODO: implement a more general version

    struct rope_args_fp16* args = (struct rope_args_fp16* ) void_args;
    int head_size = args->head_size;
    int dim = args->dim;
    int kv_dim = args->kv_dim;
    int pos = args->pos;

    int id = pi_core_id();

    int head_dim = (id*2) % head_size;
    fp16 freq = 1.0f / fastexp_gist_fp16(9.21034037198 * head_dim / (float)head_size);
    // fp16 freq = 1.0f / powf(10000.0f, head_dim/ (float)head_size);
    
    fp16 val = pos*freq;
    fp16 fcr, fci;

    if(pos <= 200){
        fcr = (fp16) cosf((float) val);
        fci = (fp16) sinf((float) val);
    } else
       cordic_cos_sin_fp16(val, &fcr, &fci);

    for(int i=id*2; i < dim ; i+=2*NUM_CORES){
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                fp16* vec = v == 0 ? args->q : args->k; // the vector to rotate (query or key)
                fp16 v0 = vec[i];
                fp16 v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
    }
}


fp16* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
  
    // copy the token embedding into x
    fp16* content_row = w->token_embedding_table + token * dim;

    fp16* x = BUFF1;
    
    // memory transfer from the token embedding table to the x vector (BUFF1)
    pi_cl_dma_copy_t token_emb_table_to_x;
    token_emb_table_to_x.ext = (uint32_t) content_row;
    token_emb_table_to_x.loc = (uint32_t) x;
    token_emb_table_to_x.size = dim*sizeof(*x);
    token_emb_table_to_x.dir = PI_CL_DMA_DIR_EXT2LOC;
    pi_cl_dma_memcpy(&token_emb_table_to_x);

    // transfer the rmsnorm weights
    pi_cl_dma_copy_t rms_weight;
    rms_weight.ext = (uint32_t) w->rms_att_weight;
    rms_weight.loc = (uint32_t) BUFF4;
    rms_weight.size = dim* sizeof(*w->rms_att_weight);
    rms_weight.dir = PI_CL_DMA_DIR_EXT2LOC;
    pi_cl_dma_memcpy(&rms_weight);       
 

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // key and value point to the k cache
        int loff = l * STEPS * kv_dim; // kv cache layer offset for convenience
        // s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
        s->xb = BUFF2;
        s->q = BUFF3;

        // transfer the weights for the v matmul
        pi_cl_dma_copy_t kv_weight;
        kv_weight.ext = (uint32_t) (w->wv + l*dim*kv_dim);
        kv_weight.loc = (uint32_t) BUFF_W_2;
        kv_weight.size = dim*kv_dim*sizeof(*w->wv);
        kv_weight.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&kv_weight);

        pi_cl_dma_wait(&token_emb_table_to_x);
        pi_cl_dma_wait(&rms_weight);

        rmsnorm_parallelized_fp16(s->xb, x, BUFF4, buffer_n_cores, dim);

        // qkv matmuls for this position

        // transfer the weights for the q matmul
        pi_cl_dma_copy_t q_weight;
        q_weight.ext = (uint32_t) (w->wq + l*dim*dim);
        q_weight.loc = (uint32_t) BUFF_W_1;
        q_weight.size = dim*dim*sizeof(*w->wq);
        q_weight.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&q_weight);
        
        pi_cl_dma_wait(&kv_weight);

        matmul(BUFF4, s->xb, BUFF_W_2, dim, kv_dim);

        // transfer the weights for the k matmul
        kv_weight.ext = (uint32_t) (w->wk + l*dim*kv_dim);
        pi_cl_dma_memcpy(&kv_weight);

        // transfer the v vector to the value cache
        pi_cl_dma_copy_t kv_to_L2;
        kv_to_L2.ext = (uint32_t) s->v;
        kv_to_L2.loc = (uint32_t) BUFF4;
        kv_to_L2.size = kv_dim*sizeof(*s->v);
        kv_to_L2.dir = PI_CL_DMA_DIR_LOC2EXT;
        pi_cl_dma_memcpy(&kv_to_L2);

        pi_cl_dma_wait(&q_weight);
        
        matmul(s->q, s->xb, BUFF_W_1, dim, dim);
        
        // transfer the key cache to BUFF_W_1 (except for the current position)
        pi_cl_dma_copy_t k_cache_to_L1;
        k_cache_to_L1.ext = (uint32_t) (s->key_cache + loff);
        k_cache_to_L1.loc = (uint32_t) BUFF_W_1;
        k_cache_to_L1.size = kv_dim * pos * sizeof(*s->key_cache);
        k_cache_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&k_cache_to_L1);

        s->k = BUFF_W_1 + kv_dim*pos;
        pi_cl_dma_wait(&kv_weight);

        matmul(s->k, s->xb, BUFF_W_2, dim, kv_dim);

        // transfer the value cache to BUFF_W_2
        pi_cl_dma_wait(&kv_to_L2);
        pi_cl_dma_copy_t v_cache_to_L1;
        v_cache_to_L1.ext = (uint32_t) (s->value_cache + loff);
        v_cache_to_L1.loc = (uint32_t) BUFF_W_2;
        v_cache_to_L1.size = kv_dim * (pos+1) * sizeof(*s->value_cache);
        v_cache_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&v_cache_to_L1);
        

        // RoPE relative positional encoding: complex-valued rotate q and k in each head

        if( head_size == NUM_CORES ){
            // current version of rope_parallelized_fp16_cl work only if for head_size == N_CORES
            // TODO: implement a more general version of rope_parallelized_fp16_cl
            struct rope_args_fp16 ra;
            ra.q = s->q;
            ra.k = s->k;
            ra.dim = dim;
            ra.head_size = head_size;
            ra.pos = pos;
            ra.kv_dim = kv_dim;

            pi_cl_team_fork(NUM_CORES, rope_parallelized_fp16_cl, &ra);

        } else {
            for (int i = 0; i < dim; i+=2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    fp16* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                    fp16 v0 = vec[i];
                    fp16 v1 = vec[i+1];
                    vec[i]   = v0 * fcr - v1 * fci;
                    vec[i+1] = v0 * fci + v1 * fcr;
                }
            }
        }

        // transfer the k vector to the key cache
        kv_to_L2.loc = (uint32_t) s->k;
        kv_to_L2.ext = (uint32_t) (s->key_cache + loff + pos * kv_dim);
        pi_cl_dma_memcpy(&kv_to_L2);

        // multihead attention

        struct llama2_mhsa_args_fp16 mhsa_args;

        mhsa_args.q = s->q;         // BUFF3
        mhsa_args.att = BUFF4;
        mhsa_args.key_cache = BUFF_W_1;
        mhsa_args.value_cache = BUFF_W_2;
        mhsa_args.xb = s->xb;       // BUFF2
        mhsa_args.pos = pos;
        mhsa_args.kv_dim = kv_dim;
        mhsa_args.kv_mul = kv_mul;
        mhsa_args.head_size = head_size;
        mhsa_args.n_heads = p->n_heads;
        mhsa_args.steps = STEPS;

        pi_cl_dma_wait(&k_cache_to_L1);
        pi_cl_dma_wait(&v_cache_to_L1);

        pi_cl_team_fork(NUM_CORES, llama2_mhsa_fp16_cl, &mhsa_args);

        pi_cl_dma_wait(&kv_to_L2);
        
        // tranfers the weights for the wo matmul
        pi_cl_dma_copy_t wo_to_L1;
        wo_to_L1.loc = (uint32_t) BUFF_W_1;
        wo_to_L1.ext = (uint32_t) (w->wo + l*dim*dim);
        wo_to_L1.size = dim * dim * sizeof(*w->wo);
        wo_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&wo_to_L1);
        
        s->xb2 = BUFF3;

        // transfer the weights for the ffn rmsnorm
        pi_cl_dma_copy_t rms_ffn_weight_to_L1;
        rms_ffn_weight_to_L1.loc = (uint32_t) BUFF4;
        rms_ffn_weight_to_L1.ext = (uint32_t) (w->rms_ffn_weight + l*dim);
        rms_ffn_weight_to_L1.size = dim * sizeof(*w->rms_ffn_weight);
        rms_ffn_weight_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&rms_ffn_weight_to_L1);

        // tranfers the weights for the first matmul in ffn
        pi_cl_dma_copy_t mm1_ffn_weight_to_L1;
        mm1_ffn_weight_to_L1.loc = (uint32_t) BUFF_W_2;
        mm1_ffn_weight_to_L1.ext = (uint32_t) (w->w1 + l*dim*hidden_dim);
        mm1_ffn_weight_to_L1.size = dim * hidden_dim * sizeof(*w->w1);
        mm1_ffn_weight_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&mm1_ffn_weight_to_L1);

        // final matmul to get the output of the attention
        pi_cl_dma_wait(&wo_to_L1);

        matmul(s->xb2, s->xb, BUFF_W_1, dim, dim);

        // residual connection back into x
        struct vect_sum_args_fp16 vsa;
        vsa.op_1 = s->xb2;          // BUFF3
        vsa.op_2 = x;               // BUFF1
        vsa.dest = x;               // BUFF1
        vsa.size = dim;

        pi_cl_team_fork(NUM_CORES, vect_sum_fp16, &vsa);
        
        // ffn rmsnorm
        pi_cl_dma_wait(&rms_ffn_weight_to_L1);
        rmsnorm_parallelized_fp16(s->xb, x, BUFF4, buffer_n_cores, dim);

        // original code for the FFN matmul:
        // matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        // matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        
        s->hb = BUFF3;
        s->hb2 = BUFF4;

        // tranfers the weights for the second matmul in ffn
        pi_cl_dma_copy_t mm2_ffn_weight_to_L1;
        mm2_ffn_weight_to_L1.loc = (uint32_t) BUFF_W_1;
        mm2_ffn_weight_to_L1.ext = (uint32_t) (w->w3 + l*dim*hidden_dim);
        mm2_ffn_weight_to_L1.size = dim * hidden_dim * sizeof(*w->w3);
        mm2_ffn_weight_to_L1.dir = PI_CL_DMA_DIR_EXT2LOC;
        pi_cl_dma_memcpy(&mm2_ffn_weight_to_L1);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        pi_cl_dma_wait(&mm1_ffn_weight_to_L1);

        matmul(s->hb, s->xb, BUFF_W_2, dim, hidden_dim);

        // transfer the weights for the third matmul in ffn
        mm1_ffn_weight_to_L1.ext = (uint32_t) (w->w2 + l*dim*hidden_dim);
        pi_cl_dma_memcpy(&mm1_ffn_weight_to_L1);

        pi_cl_dma_wait(&mm2_ffn_weight_to_L1);

        matmul(s->hb2, s->xb, BUFF_W_1, dim, hidden_dim);
        
        // SwiGLU non-linearity
        struct swiglu_args_fp16 sa;
        sa.in1 = s->hb;             // BUFF3
        sa.in2 = s->hb2;            // BUFF4
        sa.out = s->hb;             // BUFF3
        sa.dim = hidden_dim;

        pi_cl_team_fork(NUM_CORES, pulp_swiglu_fp16_cl, &sa);
        
        // transfer weights for the next layer RMSNorm or for final RMSNorm
        if(l < p->n_layers - 1)
            rms_weight.ext = (uint32_t) (w->rms_att_weight + (l+1)*dim);
        else
            rms_weight.ext = (uint32_t) (w->rms_final_weight);
        pi_cl_dma_memcpy(&rms_weight);

        // final matmul to get the output of the ffn
        pi_cl_dma_wait(&mm1_ffn_weight_to_L1);

        matmul(s->xb, s->hb, BUFF_W_2, hidden_dim, dim);
        // matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        vsa.op_1 = s->xb;         // BUFF2
        vsa.op_2 = x;             // BUFF1
        vsa.dest = x;             // BUFF1
        vsa.size = dim;

        pi_cl_team_fork(NUM_CORES, vect_sum_fp16, &vsa);
    }
    
    int mm_div = 4;   // split matmul in mm_div part because it's too big. Must be a divider of vocab_size
    int part = p->vocab_size / mm_div; 
    s->logits = BUFF4;
    
    pi_cl_dma_copy_t mm_weights_to_BUFF_W_1, mm_weights_to_BUFF_W_2;
    mm_weights_to_BUFF_W_1.ext = (uint32_t) w->wcls;
    mm_weights_to_BUFF_W_1.loc = (uint32_t) BUFF_W_1;
	mm_weights_to_BUFF_W_1.size = dim * part * sizeof(*w->wcls);
	mm_weights_to_BUFF_W_1.dir = PI_CL_DMA_DIR_EXT2LOC;
    pi_cl_dma_memcpy(&mm_weights_to_BUFF_W_1);

    mm_weights_to_BUFF_W_2.loc = (uint32_t) BUFF_W_2;
    mm_weights_to_BUFF_W_2.size = dim * part * sizeof(*w->wcls);
    mm_weights_to_BUFF_W_2.dir = PI_CL_DMA_DIR_EXT2LOC;
    
    // final rmsnorm
    pi_cl_dma_wait(&rms_weight);

    rmsnorm_parallelized_fp16(s->xb, x, BUFF4, buffer_n_cores, dim);
    
    // classifier into logits. Orignal implementation: 
    // matmul(s->logits, s->xb, w->wcls, p->dim, p->vocab_size);

    for(int i=0; i<mm_div; i+=2){
        mm_weights_to_BUFF_W_2.ext = (uint32_t) (w->wcls + (i+1)*part*dim);
        pi_cl_dma_memcpy(&mm_weights_to_BUFF_W_2);

        pi_cl_dma_wait(&mm_weights_to_BUFF_W_1);
        matmul(s->logits+i*part, s->xb, BUFF_W_1, p->dim, part);

        if(i < mm_div - 2){
            mm_weights_to_BUFF_W_1.ext = (uint32_t) (w->wcls + (i+2)*part*dim);
            pi_cl_dma_memcpy(&mm_weights_to_BUFF_W_1);
        }
        
        pi_cl_dma_wait(&mm_weights_to_BUFF_W_2);
        matmul(s->logits+(i+1)*part, s->xb, BUFF_W_2, p->dim, part);
    }

    return s->logits;
}


// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
#ifndef _TokenIndex_
#define _TokenIndex_
typedef struct {
    char *str;
    int id;
} TokenIndex;
#endif

typedef struct {
    char** vocab;
    fp16* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

void build_tokenizer(Tokenizer* t, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = VOCAB;
    t->vocab_scores = VOCAB_SCORES;
    t->sorted_vocab = SORTED_VOCAB;
    
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    t->max_token_length = MAX_TOKEN_LENGTH;
    int len;
    int j=0;
    for (int i = 0; i < vocab_size; i++) {
        t->vocab[i] = &VOCAB_DATA[j];
        while(VOCAB_DATA[j] != '\0' && i < vocab_size-1)
            j++;
        j++;
    }
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    /*
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    */
   // Instead of sscanf, we can use a simple if statement to check if the token is a byte token
    if(piece[0]=='<' && piece[1] == '0' && piece[2]=='x' && piece[5]=='>'){
        int cifra1, cifra2;
        if('0' <= piece[3] && piece[3]<= '9')
            cifra1 = piece[3] - '0';
        else
            cifra1 = piece[3] - 'A' + 10; 
        if('0' <= piece[4] && piece[4] <= '9')
            cifra2 = piece[4] - '0';
        else
            cifra2 = piece[4] - 'A' + 10;
        byte_val = cifra1*16 + cifra2;

        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void *bsearch (const void *key, const void *base0, size_t nmemb, size_t size, int (*compar)(const void *, const void *))
{
	const char *base = (const char *) base0;
	int lim, cmp;
	const void *p;

	for (lim = nmemb; lim != 0; lim >>= 1) {
		p = base + (lim >> 1) * size;
		cmp = (*compar)(key, p);
		if (cmp == 0)
			return (void *)p;
		if (cmp > 0) {	/* key > p: move right */
			base = (const char *)p + size;
			lim--;
		} /* else move left */
	}
	return (NULL);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL)
        exit(1);
    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    char* str_buffer = (char* ) BUFF1;
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        fp16 best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        
        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
#ifndef _ProbIndex_
#define _ProbIndex_
typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling
#endif

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(fp16* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    fp16 max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(fp16* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    fp16 cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int part_probIndex(ProbIndex* a, int l, int h){
    float p = a[h].prob;
    int i = l-1;
    ProbIndex tmp;
    for(int j=l;j<h;j++){
        if(a[j].prob>=p){
            i++;
            tmp = a[j];
            a[j] = a[i];
            a[i] = tmp;
        }
    }
    tmp = a[i+1];
    a[i+1] = a[h];
    a[h] = tmp;
    return i+1;
}

void quickSort_probIndex(ProbIndex* a, int l, int h){
    if(l < h){
        int p = part_probIndex(a, l, h);
        quickSort_probIndex(a, l, p-1);
        quickSort_probIndex(a, p+1, h);
    }
}

int sample_topp(fp16* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const fp16 cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    quickSort_probIndex(probindex, 0, n0-1);
    
    // truncate the list where cumulative probability exceeds topp
    fp16 cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    fp16 r = coin * cumulative_prob;
    fp16 cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex*)PROB_INDEX;
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, fp16* logits, char isLastPos) {
    // sample the token given the logits and some hyperparameters
    int next;
    //printf("sampler->temperature: %f\n", sampler->temperature);
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++)
            logits[q] /= sampler->temperature; 

        // apply softmax to the logits to get the probabilities for next token
        pulp_vector_softmax_fp16(logits, logits, buffer_n_cores, sampler->vocab_size);

        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

void net_step(){
    INIT_STATS();
    PRE_START_STATS();
    START_STATS();

    int steps = STEPS;
    float temperature = TEMPERATURE;
    float topp = 0.9f;
    unsigned long long rng_seed = RND_SEED;

    Transformer transformer;
    build_transformer(&transformer);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    fp16* log;
    int token, next, tok_diversi=0;
    int num_prompt_tokens=0;
    int* prompt_tokens = PROMPT_TOKENS;

    encode(&tokenizer, PROMPT, 1, 0, prompt_tokens, &num_prompt_tokens);

    token = prompt_tokens[0];
    for(int pos = 0; pos < steps; pos++ ) {

        log = forward(&transformer, token, pos);
        
        if(pos < num_prompt_tokens -1)
            next = prompt_tokens[pos+1];
        else{
            next = sample(&sampler, log, pos==STEPS-1);
        }
        
        if(next==1)
            break; 
        
        char* piece = decode(&tokenizer, token, next);
        
        safe_printf(piece);

        token = next;
    }

    STOP_STATS();
    return;
}