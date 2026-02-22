# filename: vae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(12, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def training_step(self, x):
        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar)
        loss.backward()
        return loss

# Example usage
# model = VAE(input_dim=784, latent_dim=20)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# for epoch in range(10):
#     for data in dataloader:
#         optimizer.zero_grad()
#         loss = model.training_step(data)
#         optimizer.step()
#         print('Epoch [' + str(epoch+1) + '/' + str(10) + '], Loss: ' + str(loss.item()) + '.4f')