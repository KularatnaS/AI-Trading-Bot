import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, seq_len, d_model, N, d_ff, dropout):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(seq_len * d_model, d_ff)
        self.norm1 = nn.BatchNorm1d(d_ff)
        self.linears = nn.ModuleList([nn.Linear(d_ff, d_ff) for _ in range(N)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(d_ff) for _ in range(N)])
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, latent_dims)
        # self.linear3 = nn.Linear(d_ff, latent_dims)

        # self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        # self.kl = 0

    def forward(self, x):
        x = F.relu(self.norm1(self.linear1(x)))
        for linear, norm in zip(self.linears, self.norms):
            x = F.relu(norm(linear(x)))
            x = self.dropout(x)
        mu = self.linear2(x)
        # sigma = torch.exp(self.linear3(x))
        # z = mu + sigma*self.N.sample(mu.shape)
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return mu


class DecoderVAE(nn.Module):
    def __init__(self, latent_dims, seq_len, d_model, N, d_ff, dropout):
        super(DecoderVAE, self).__init__()
        self.linear1 = nn.Linear(latent_dims, d_ff)
        self.norm1 = nn.BatchNorm1d(d_ff)
        self.linears = nn.ModuleList([nn.Linear(d_ff, d_ff) for _ in range(N)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(d_ff) for _ in range(N)])
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, seq_len * d_model)

    def forward(self, x):
        x = F.relu(self.norm1(self.linear1(x)))
        for linear, norm in zip(self.linears, self.norms):
            x = F.relu(norm(linear(x)))
            x = self.dropout(x)
        x = self.linear2(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dims, d_model, seq_len, N, d_ff, dropout):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, seq_len, d_model, N, d_ff, dropout)
        self.decoder = DecoderVAE(latent_dims, seq_len, d_model, N, d_ff, dropout)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class ClassificationModel(nn.Module):
    def __init__(self, n_trade_decisions, seq_len, d_model, N, d_ff, dropout):
        super(ClassificationModel, self).__init__()
        self.linear1 = nn.Linear(seq_len * d_model, d_ff)
        self.norm1 = nn.BatchNorm1d(d_ff)
        self.linears = nn.ModuleList([nn.Linear(d_ff, d_ff) for _ in range(N)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(d_ff) for _ in range(N)])
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, n_trade_decisions)

    def forward(self, x):
        x = F.relu(self.norm1(self.linear1(x)))
        for linear, norm in zip(self.linears, self.norms):
            x = F.relu(norm(linear(x)))
            x = self.dropout(x)
        mu = self.linear2(x)
        return mu
