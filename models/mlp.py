from flax import nn


class MLP(nn.Module):
    def apply(self, x, hidden_size, output_size):
        z = nn.Dense(x, hidden_size)
        h = nn.relu(z)
        h = nn.Dense(h, hidden_size)
        h = nn.relu(h)
        y = nn.Dense(h, output_size)
        return y
