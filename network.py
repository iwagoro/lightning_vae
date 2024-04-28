import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
    
class depthwise_separable_conv_transpose(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(depthwise_separable_conv_transpose, self).__init__()
        self.depthwise = nn.ConvTranspose2d(
            nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias 
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            depthwise_separable_conv(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            depthwise_separable_conv(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            depthwise_separable_conv(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            depthwise_separable_conv(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.mu = nn.Linear(512,2)
        self.dev = nn.Linear(512,2)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        dev = self.dev(x)
        ep = torch.randn_like(mu)
        z = mu + torch.exp(dev / 2) * ep
        return z,mu,dev


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.linear = nn.Linear(2,512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 512, 1, 1)
        x = self.decoder(x)
        
        return x


def vae_loss(x, x_hat, mu, log_var):
    # Reconstruction loss (MSE)
    reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
    # KL divergence loss
    kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Total VAE loss
    vae_loss = reconstruction_loss + kl_divergence_loss
    return vae_loss

class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent,mu,dev = self.encoder(x)
        self.mu = mu
        self.dev = dev
        x = self.decoder(latent)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)  # 生成された画像
        loss = vae_loss(x, x_hat, self.mu, self.dev)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self,batch,batch_idx):
        
        example_img,_ = batch
        predicted_img = self.forward(example_img)
        self.logger.experiment.add_image("generated_images", predicted_img[0], self.current_epoch)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)