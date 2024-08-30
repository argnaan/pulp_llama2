# PULP-Llama2
Adaptation of [Llama2.c](https://github.com/karpathy/llama2.c) to run on a [PULP](https://github.com/pulp-platform/pulp) microcontroller.

## How to try it

Clone this project

```bash
git clone https://github.com/argnaan/pulp_llama2.git
```

Get PULP-TrainLib from its github [repository](https://github.com/pulp-platform/pulp-trainlib)

```bash
git clone https://github.com/pulp-platform/pulp-trainlib.git
```
And modify Makefile with the correct path to the Train-Lib

Generate the header file which contains the configuration and the weights of the model:
```
make get_header
``` 

It can be runned thorugh [GVSoC](https://github.com/pulp-platform/GvSoc), a PULP chip simulator, part of the [pulp-sdk](https://github.com/pulp-platform/pulp-sdk).
```
make clean all run
```


Look inside the genHeader.c file to modify some option such as the initial prompt or the number of steps, then regenerate the header.
You can also try different model in the format of the [Llama2.c](https://github.com/karpathy/llama2.c) project.
