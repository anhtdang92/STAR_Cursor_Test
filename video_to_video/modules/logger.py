import torch
import torchvision
import numpy as np
from einops import rearrange

class ImageLogger:
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True):
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = True
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.logger_log_images and pl_module.global_step % self.batch_freq == 0 and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **pl_module.log_images_cond)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            # log images
            for k in images:
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].permute(0, 2, 3, 1).numpy()
                    images[k] = (images[k] + 1.0) / 2.0
                    images[k] = (images[k] * 255).astype(np.uint8)
                    images[k] = images[k].transpose(0, 3, 1, 2)
                    images[k] = torch.from_numpy(images[k])

            self.log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def log_images(self, pl_module, images, global_step, split="train"):
        for k in images:
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)

            # log images
            for k in images:
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].permute(0, 2, 3, 1).numpy()
                    images[k] = (images[k] + 1.0) / 2.0
                    images[k] = (images[k] * 255).astype(np.uint8)
                    images[k] = images[k].transpose(0, 3, 1, 2)
                    images[k] = torch.from_numpy(images[k])

            self.log_images(pl_module, images, global_step, split) 