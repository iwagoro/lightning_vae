import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from dsconv import depthwise_separable_conv
from spconv import sub_pixel_conv



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            depthwise_separable_conv(1, 32, kernel_size=3, padding=1), #16
            nn.ReLU(),
            nn.MaxPool2d(2),
            depthwise_separable_conv(32, 64, kernel_size=3, padding=1),# 8
            nn.ReLU(),
            nn.MaxPool2d(2),
            depthwise_separable_conv(64, 128, kernel_size=3, padding=1), #4
            nn.ReLU(),
            nn.MaxPool2d(2),
            depthwise_separable_conv(128, 256, kernel_size=3, padding=1),# 2
            nn.ReLU(),
            nn.MaxPool2d(2),
            depthwise_separable_conv(256, 512, kernel_size=3, padding=1),# 1
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
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.linear = nn.Linear(2,512)
        self.decoder = nn.Sequential(
            sub_pixel_conv(512,256,2),#2
            nn.ReLU(),
            sub_pixel_conv(256,128,2),#4
            nn.ReLU(),
            sub_pixel_conv(128,64,2),#8
            nn.ReLU(),
            sub_pixel_conv(64,32,2),#16
            nn.ReLU(),
            sub_pixel_conv(32,16,2),#32
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.pointwise = nn.Conv2d(16,1,1)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1,512,1,1)
        x = self.decoder(x)
        x = self.pointwise(x)
        return x


def vae_loss(x, x_hat, mu, log_var):
    # Reconstruction loss (MSE)
    reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
    # KL divergence loss
    kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Total VAE loss
    vae_loss = reconstruction_loss + kl_divergence_loss
    return vae_loss

class AutoEncoder(LightningModule):
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
        # loss = vae_loss(x, x_hat, self.mu, self.dev)
        loss = nn.MSELoss()(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self,batch,batch_idx):
        
        example_img,_ = batch
        predicted_img = self.forward(example_img)
        # loss = vae_loss(example_img, predicted_img, self.mu, self.dev)
        loss = nn.MSELoss()(example_img,predicted_img)
        self.log("val_loss", loss, prog_bar=True)
        self.logger.experiment.add_image("generated_images", predicted_img[0], self.current_epoch)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)