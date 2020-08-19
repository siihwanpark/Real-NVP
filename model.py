import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (D-d))
        )
        
    def forward(self, x_a, x_b):
        output = self.layers(x_a) # [batch, 2 * (D-d)]
        log_s, t = output[:, :self.D-self.d], output[:, self.D-self.d:]

        z_a = x_a # [batch, d]
        z_b = torch.exp(log_s) * x_b + t # [batch, D-d]

        return z_a, z_b, log_s

    def backward(self, z_a, z_b):
        output = self.layers(z_a)
        log_s, t = output[:, :self.D-self.d], output[:, self.D-self.d:]

        x_a = z_a
        x_b = torch.exp(-log_s) * (z_b - t)

        return x_a, x_b

class RealNVP(nn.Module):
    def __init__(self, k, d, D, hidden_dim):
        super(RealNVP, self).__init__()
        self.k = k
        self.d = d
        self.D = D
        self.layers = nn.ModuleList(AffineCouplingLayer(d, D, hidden_dim) for _ in range(k))

    def forward(self, x):
        x_a, x_b = x[:, :self.d], x[:, self.d:]
        h_a, h_b = x_a, x_b
        log_det_Jacobian = 0

        for i in range(self.k):
            if i % 2 == 0:
                h_a, h_b, log_s = self.layers[i](h_a, h_b)
            else:
                h_b, h_a, log_s = self.layers[i](h_b, h_a)

            log_det_Jacobian += log_s.sum(dim = -1).mean()

        return torch.cat([h_a, h_b], dim = -1), log_det_Jacobian

    def backward(self, z):
        z_a, z_b = z[:, :self.d], z[:, self.d:]
        h_a, h_b = z_a, z_b

        for i in reversed(range(self.k)):
            if i % 2 == 0:
                h_a, h_b = self.layers[i].backward(h_a, h_b)
            else:
                h_b, h_a = self.layers[i].backward(h_b, h_a)

        return torch.cat([h_a, h_b], dim = -1)