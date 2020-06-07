"""
Generate a random Sinusoid and fit to it using FLAX.
"""
import time

import numpy as onp
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random
from flax import nn, optim

from models import MLP


# generate data
N = 200
X = onp.linspace(-3, 3, N)[:, None]
true_Y = 2.5 * onp.sin(X + 0.5)
Y = true_Y + onp.random.randn(N, 1)

module = MLP.partial(hidden_size=32, output_size=1)
_, params = module.init(random.PRNGKey(0), X)
model = nn.Model(module, params)


@jax.jit
def train_step(optimizer, batch):
    def loss_fn(model):
        X, Y = batch
        pred = model.module.call(model.params, X)
        return jnp.square(Y - pred).mean()

    # optimizer.target is the flax.nn.Model object
    loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
    # `apply_gradient` returns a new `Optimizer` instance with the updated target and optimizer state.
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss


optimizer_def = optim.Momentum(learning_rate=0.01, beta=0.9)
optimizer = optimizer_def.create(model)
train_steps = 1000

before = time.time()
for i in range(train_steps):
    optimizer, loss = train_step(optimizer, (X, Y))
    print(loss)

elapsed = time.time() - before
print(f"mean square error: {loss}, elapsed: {elapsed}")
trained_model = optimizer.target

# Plot predictions after training
plt.scatter(X, Y)
plt.plot(X, true_Y)
plt.plot(X, trained_model(X))  # model prediction
plt.show()
