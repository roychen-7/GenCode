import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc2(h)

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2)
        )
    
    def forward(self, z):
        return self.net(z)

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lookback_window = config["lookback_window"]
        self.latent_dim = config.get("latent_dim", 10)
        self.hidden_dim = config.get("hidden_dim", 128)
        
        input_dim = self.lookback_window
        
        self.encoder = Encoder(input_dim, self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim, 1)
        self.discriminator = Discriminator(self.latent_dim)
        
        self.gamma = config.get("gamma", 6.0)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x[:, :, 0]
        
        mu, logvar = self.encoder(x_flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        
        return recon
    
    def compute_vae_loss(self, x, recon, mu, logvar):
        x_target = x[:, -1, 0:1]
        recon_loss = F.mse_loss(recon, x_target, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_loss, recon_loss, kl_loss

def compute_loss(preds, labels):
    return F.mse_loss(preds, labels)
