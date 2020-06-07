# MAML implementation in Flax
Model Agnostic Meta Learning (MAML) implemented in Flax, the neural network library for JAX.

## Introduction 
This repository implements a MAML example for sinusoid regression in Flax. The idea of MAML is to learn the initial weight values of a model that can quickly adapt to new tasks. For more information, check the [paper](https://arxiv.org/abs/1703.03400).

This implementation uses only default Flax components like `flax.nn.Model` and `flax.nn.Module`, showing that this kind of optimization-based Meta Learning algorithms can easily be implemented in Flax/JAX.

It is based on the [MAML implementation in JAX by Eric Jang](https://blog.evjang.com/2019/02/maml-jax.html) and updated to use Flax components. I have only implemented the sinusoid example so far, but I intend to add the Omniglot example too.

There is also an implementation of a model that fits just to one sinusoid, without meta learning, useful to see the difference between the two approaches. This approach is implemented in `main_wo_maml.py`.

## Running
Just run `python main.py` to train MAML for fast adaptation to sinusoid regression tasks.


## Citation
If you use this code in your work please cite the original paper:
```
@inproceedings{finn2017model,
  title={Model-agnostic meta-learning for fast adaptation of deep networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={Proceedings of the 34th International Conference on Machine Learning-Volume 70},
  pages={1126--1135},
  year={2017},
  organization={JMLR. org}
}
```
