"""
MAML implementation in Flax and JAX that fits to random sinusoids.

It uses a default flax.nn.Model everywhere.
"""

import time
from functools import partial  # to use with vmap

import numpy as onp
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random
from flax import nn, optim

from models import MLP
from datasets import generate_sinusoids, generate_sin_tasks
from utils import plot_sinusoid


def create_model():
    # Adapted from examples/imagenet/train.py
    input_shape = (1, 1)
    module = MLP.partial(hidden_size=40, output_size=1)
    _, params = module.init_by_shape(random.PRNGKey(0), [input_shape])
    model = nn.Model(module, params)
    return model


def main():
    batch_size = 25
    n_points = 10
    num_steps = 70000
    lr = 0.001
    inner_lr = 0.01
    inner_steps = 1

    model = create_model()
    optimizer_def = optim.Adam(learning_rate=lr)
    optimizer = optimizer_def.create(model)
    inner_optimizer_def = optim.GradientDescent(learning_rate=inner_lr)

    @jax.jit
    def task_loss(model, batch):
        X, Y = batch
        pred = model(X)
        return jnp.square(Y - pred).mean()

    @jax.jit
    def maml_fit_task(model, batch):
        model_grad = jax.grad(task_loss)(model, batch)
        inner_opt = inner_optimizer_def.create(model)
        for _ in range(inner_steps):  # do it for k steps
            inner_opt = inner_opt.apply_gradient(model_grad)
        return inner_opt.target

    @jax.jit
    def train_step(optimizer, batch):
        # optimizer.target is the flax.nn.Model object
        model = optimizer.target

        @jax.jit
        def maml_loss(model, train_x, train_y, val_x, val_y):
            train_batch = (train_x, train_y)
            val_batch = (val_x, val_y)
            # fit model to the task (inner loop optimization)
            updated_model = maml_fit_task(model, train_batch)
            # after inner loop, evaluate model on the meta val task data
            loss = task_loss(updated_model, val_batch)
            return loss

        @jax.jit
        def loss_fn(model):
            train_x, train_y, val_x, val_y = batch
            task_losses = jax.vmap(partial(maml_loss, model))(
                train_x, train_y, val_x, val_y
            )
            return jnp.mean(task_losses)

        loss, grad = jax.value_and_grad(loss_fn)(model)
        # `apply_gradient` returns a new `Optimizer` instance with the updated target and optimizer state.
        optimizer = optimizer.apply_gradient(grad)

        return optimizer, loss

    before = time.time()
    for i in range(num_steps):
        # (meta_train_x, meta_train_y, meta_val_x, meta_val_y)
        batch = generate_sin_tasks(batch_size, n_points)
        optimizer, loss = train_step(optimizer, batch)
        if i % 500 == 0:
            print(f"Iteration {i}: loss: {loss}")

    elapsed = time.time() - before
    print(f"mean square error: {loss}, elapsed: {elapsed}")

    # fit a random sinusoid and plot the result
    data, labels, amp, phase = generate_sinusoids(1, n_points)
    task_model = maml_fit_task(optimizer.target, (data, labels))
    x = onp.arange(-5, 5, 0.1).reshape((-1, 1))
    y = task_model(x)
    plot_sinusoid(
        x.squeeze(), y.squeeze(), amp[0].squeeze().item(), phase[0].squeeze().item()
    )


if __name__ == "__main__":
    main()
