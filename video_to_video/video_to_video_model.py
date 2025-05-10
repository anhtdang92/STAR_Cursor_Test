import os
import os.path as osp
import random
from typing import Any, Dict

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.nn as nn
from safetensors.torch import load_file

from video_to_video.modules.autoencoder import AutoencoderKLTemporalDecoder
from video_to_video.utils.config import cfg
from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.utils.logger import get_logger

logger = get_logger()

class VideoToVideo_sr(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        
        # Initialize autoencoder
        autoencoder_config = {
            'ch': 128,
            'out_ch': 3,
            'ch_mult': (1, 2, 4, 8),
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.0,
            'resamp_with_conv': True,
            'in_channels': 3,
            'resolution': 256,
            'z_channels': 4,
            'double_z': True,
            'use_linear_attn': False,
            'attn_type': 'vanilla',
            'loss': {
                'target': 'video_to_video.modules.losses.LPIPSWithDiscriminator',
                'params': {
                    'disc_start': 50001,
                    'kl_weight': 0.000001,
                    'disc_weight': 0.5
                }
            }
        }
        self.autoencoder = AutoencoderKLTemporalDecoder(
            ddconfig=autoencoder_config,
            embed_dim=4,
            ckpt_path=None
        )
        
        # Initialize diffusion model
        self.diffusion = GaussianDiffusion(
            denoise_fn=self.autoencoder,
            schedule=noise_schedule(timesteps=1000),
            timesteps=1000,
            loss_type='l2'
        )
        
        # Load model weights with better error handling
        try:
            # First try loading as safetensors
            if cfg.model_path.endswith('.safetensors'):
                state_dict = load_file(cfg.model_path)
                self.load_state_dict(state_dict)
            else:
                # Try loading as PyTorch checkpoint
                try:
                    # First try with weights_only=True
                    try:
                        load_dict = torch.load(cfg.model_path, map_location='cpu', weights_only=True)
                    except Exception as e:
                        logger.warning(f"Failed to load with weights_only=True, trying without: {e}")
                        load_dict = torch.load(cfg.model_path, map_location='cpu')
                    
                    if isinstance(load_dict, dict) and 'state_dict' in load_dict:
                        self.load_state_dict(load_dict['state_dict'])
                    else:
                        self.load_state_dict(load_dict)
                except Exception as e:
                    logger.error(f"Error loading model as PyTorch checkpoint: {e}")
                    raise
        except Exception as e:
            logger.error(f"Failed to load model from {cfg.model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
        
        # Freeze autoencoder parameters
        for param in self.autoencoder.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.diffusion(x)
    
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a sample from the model."""
        return self.diffusion.sample(x)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input using the autoencoder."""
        return self.autoencoder.encode(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation using the autoencoder."""
        return self.autoencoder.decode(z)
    
    def get_last_layer(self) -> torch.Tensor:
        """Get the last layer of the model."""
        return self.autoencoder.get_last_layer()

    def test(self, input: Dict[str, Any], total_noise_levels=1000, \
                 steps=50, solver_mode='fast', guide_scale=7.5, max_chunk_len=32):
        video_data = input['video_data']
        y = input['y']
        (target_h, target_w) = input['target_res']

        video_data = F.interpolate(video_data, [target_h,target_w], mode='bilinear')

        logger.info(f'video_data shape: {video_data.shape}')
        frames_num, _, h, w = video_data.shape

        padding = pad_to_fit(h, w)
        video_data = F.pad(video_data, padding, 'constant', 1)

        video_data = video_data.unsqueeze(0)
        bs = 1
        video_data = video_data.to(self.device)

        video_data_feature = self.vae_encode(video_data)
        torch.cuda.empty_cache()

        y = self.text_encoder(y).detach()

        with amp.autocast(enabled=True):

            t = torch.LongTensor([total_noise_levels-1]).to(self.device)
            noised_lr = self.diffusion.diffuse(video_data_feature, t)

            model_kwargs = [{'y': y}, {'y': self.negative_y}]
            model_kwargs.append({'hint': video_data_feature})

            torch.cuda.empty_cache()
            chunk_inds = make_chunks(frames_num, interp_f_num=0, max_chunk_len=max_chunk_len) if frames_num > max_chunk_len else None

            solver = 'dpmpp_2m_sde' # 'heun' | 'dpmpp_2m_sde' 
            gen_vid = self.diffusion.sample_sr(
                noise=noised_lr,
                model=self.generator,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=0.2,
                solver=solver,
                solver_mode=solver_mode,
                return_intermediate=None,
                steps=steps,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization='trailing',
                chunk_inds=chunk_inds,)
            torch.cuda.empty_cache()

            logger.info(f'sampling, finished.')
            vid_tensor_gen = self.vae_decode_chunk(gen_vid, chunk_size=3)

            logger.info(f'temporal vae decoding, finished.')

        w1, w2, h1, h2 = padding
        vid_tensor_gen = vid_tensor_gen[:,:,h1:h+h1,w1:w+w1]

        gen_video = rearrange(
            vid_tensor_gen, '(b f) c h w -> b c f h w', b=bs)

        torch.cuda.empty_cache()
        
        return gen_video.type(torch.float32).cpu()

    def temporal_vae_decode(self, z, num_f):
        return self.vae.decode(z/self.vae.config.scaling_factor, num_frames=num_f).sample

    def vae_decode_chunk(self, z, chunk_size=3):
        z = rearrange(z, "b c f h w -> (b f) c h w")
        video = []
        for ind in range(0, z.shape[0], chunk_size):
            num_f = z[ind:ind+chunk_size].shape[0]
            video.append(self.temporal_vae_decode(z[ind:ind+chunk_size],num_f))
        video = torch.cat(video)
        return video

    def vae_encode(self, t, chunk_size=1):
        num_f = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        z_list = []
        for ind in range(0,t.shape[0],chunk_size):
            z_list.append(self.vae.encode(t[ind:ind+chunk_size]).latent_dist.sample())
        z = torch.cat(z_list, dim=0)
        z = rearrange(z, "(b f) c h w -> b c f h w", f=num_f)
        return z * self.vae.config.scaling_factor
    

def pad_to_fit(h, w):
    BEST_H, BEST_W = 720, 1280

    if h < BEST_H:
        h1, h2 = _create_pad(h, BEST_H)
    elif h == BEST_H:
        h1 = h2 = 0
    else: 
        h1 = 0
        h2 = int((h + 48) // 64 * 64) + 64 - 48 - h

    if w < BEST_W:
        w1, w2 = _create_pad(w, BEST_W)
    elif w == BEST_W:
        w1 = w2 = 0
    else:
        w1 = 0
        w2 = int(w // 64 * 64) + 64 - w
    return (w1, w2, h1, h2)

def _create_pad(h, max_len):
    h1 = int((max_len - h) // 2)
    h2 = max_len - h1 - h
    return h1, h2


def make_chunks(f_num, interp_f_num, max_chunk_len, chunk_overlap_ratio=0.5):
    MAX_CHUNK_LEN = max_chunk_len
    MAX_O_LEN = MAX_CHUNK_LEN * chunk_overlap_ratio
    chunk_len = int((MAX_CHUNK_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1)
    o_len = int((MAX_O_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1)
    chunk_inds = sliding_windows_1d(f_num, chunk_len, o_len)
    return chunk_inds


def sliding_windows_1d(length, window_size, overlap_size):
    stride = window_size - overlap_size
    ind = 0
    coords = []
    while ind<length:
        if ind+window_size*1.25>=length:
            coords.append((ind,length))
            break
        else:
            coords.append((ind,ind+window_size))
            ind += stride  
    return coords
