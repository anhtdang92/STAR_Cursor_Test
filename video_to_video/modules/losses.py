import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Dict, Any

class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start: int,
        kl_weight: float = 0.000001,
        disc_weight: float = 0.5,
        perceptual_weight: float = 1.0,
    ):
        super().__init__()
        self.disc_start = disc_start
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight
        self.perceptual_weight = perceptual_weight
        
        self.perceptual_loss = LPIPS().eval()
        self.discriminator = NLayerDiscriminator(input_nc=3)
        
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        posterior: Any,
        optimizer_idx: int,
        global_step: int,
        last_layer: Optional[torch.Tensor] = None,
        split: str = "train",
    ) -> torch.Tensor:
        if optimizer_idx == 0:
            # Reconstruction loss
            rec_loss = torch.abs(inputs - reconstructions)
            if split == "train":
                rec_loss = torch.mean(rec_loss)
            else:
                rec_loss = rec_loss.mean(dim=[1, 2, 3])
            
            # Perceptual loss
            p_loss = self.perceptual_loss(inputs, reconstructions)
            
            # KL loss
            kl_loss = posterior.kl()
            kl_loss = torch.mean(kl_loss)
            
            # Discriminator loss (if training)
            d_loss = 0.0
            if global_step >= self.disc_start and self.disc_weight > 0:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                d_loss = self.disc_weight * self.hinge_d_loss(logits_real, logits_fake)
            
            # Total loss
            loss = rec_loss + self.perceptual_weight * p_loss + self.kl_weight * kl_loss + d_loss
            
            return loss
            
        if optimizer_idx == 1:
            # Train discriminator
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            g_loss = -torch.mean(logits_fake)
            
            if split == "train":
                d_loss = self.hinge_d_loss(logits_real, logits_fake)
                return d_loss
            else:
                return g_loss
    
    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).features[:30].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_features = self.get_features(x)
        y_features = self.get_features(y)
        
        return torch.mean(torch.stack([
            torch.mean(torch.abs(x_feat - y_feat))
            for x_feat, y_feat in zip(x_features, y_features)
        ]))
        
    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in {3, 8, 15, 22, 29}:  # After each conv block
                features.append(x)
        return features

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super().__init__()
        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input) 