import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    def __init__(self, d, D, hidden_dim):
        super(AffineCouplingLayer, self).__init__()
        
        # d : dimension of first partition
        # D : dimension of x
        assert D > d, "d should be less than D."
        
        self.d = d
        self.D = D
        self.layers = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (D-d))
        )
        
    def forward(self, x_a, x_b):
        output = self.layers(x_a) # [batch, 2 * (D-d)]
        log_s, t = output[:, :self.D-self.d], output[:, self.D-self.d:]

        z_a = x_a # [batch, d]
        z_b = torch.exp(log_s) * x_b + t # [batch, D-d]

        return z_a, z_b

    def backward(self, z_a, z_b):
        output = self.layers(z_a)
        log_s, t = output[:, :self.D-self.d], output[:, self.D-self.d:]

        x_a = z_a
        x_b = torch.exp(-log_s) * (z_b - t)

        return x_a, x_b

class RealNVP(nn.Module):
    def __init__(self, d, D, hidden_dim):
        self.d = d
        self.D = D
        self.layers = nn.ModuleList(AffineCouplingLayer(d, D, hidden_dim) for _ in range(4))

    def forward(self, x):
        x_a, x_b = x[:, :self.d], x[:, self.d:]
        h1_a, h1_b = self.layers[0](x_a, x_b)
        h2_a, h2_b = self.layers[0](h1_b, h1_a)