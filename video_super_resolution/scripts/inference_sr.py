import os
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from argparse import ArgumentParser, Namespace
import json
from typing import Any, Dict, List, Mapping, Tuple
from easydict import EasyDict
from pathlib import Path

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(base_path)
from video_to_video.video_to_video_model import VideoToVideo_sr
from video_to_video.utils.seed import setup_seed
from video_to_video.utils.logger import get_logger
from video_super_resolution.color_fix import adain_color_fix

from inference_utils import *

logger = get_logger()

# Set PyTorch thread optimization
torch.set_num_threads(16)

class STAR():
    def __init__(
        self,
        output_path: str,
        model_path: str,
        solver_mode: str = 'balanced',
        steps: int = 20,
        guide_scale: float = 7.5,
        upscale: int = 4,
        max_chunk_len: int = 16,
        frame_stride: int = 8,
        resize_short_edge: int = 360,
        denoise_level: float = 0.0,
        preserve_details: bool = True,
        device: str = 'cuda',
        amp: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        batch_size: int = 1
    ):
        self.output_path = output_path
        self.model_path = model_path
        self.solver_mode = solver_mode
        self.steps = steps
        self.guide_scale = guide_scale
        self.upscale = upscale
        self.max_chunk_len = max_chunk_len
        self.frame_stride = frame_stride
        self.resize_short_edge = resize_short_edge
        self.denoise_level = denoise_level
        self.preserve_details = preserve_details
        self.device = torch.device(device)
        self.amp = amp
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size

        # Enable cuDNN benchmarking for better performance
        if self.device.type == 'cuda':
            cudnn.benchmark = True

        # Load model
        self.model = VideoToVideo_sr(EasyDict(model_path=model_path))
        self.model.to(self.device)
        self.model.eval()

    def enhance_a_video(self, input_path: str):
        # Create data loader with optimized settings
        dataloader, input_fps = self.create_dataloader(input_path)
        
        # Process video with AMP if enabled
        with autocast(enabled=self.amp):
            self.process_video(dataloader, input_fps)

    def create_dataloader(self, input_path: str):
        """Create an optimized data loader for video processing."""
        logger.info('Loading video: {}'.format(input_path))
        input_frames, input_fps = load_video(input_path)
        logger.info('Input FPS: {}'.format(input_fps))
        
        # Preprocess frames
        video_data = preprocess(input_frames)
        _, _, h, w = video_data.shape
        logger.info('Input resolution: {}'.format((h, w)))
        
        # Calculate target resolution
        target_h, target_w = h * self.upscale, w * self.upscale
        logger.info('Target resolution: {}'.format((target_h, target_w)))
        
        # Create dataset
        dataset = VideoDataset(
            video_data,
            target_res=(target_h, target_w),
            max_chunk_len=self.max_chunk_len,
            frame_stride=self.frame_stride,
            resize_short_edge=self.resize_short_edge
        )
        
        # Create data loader with optimized settings
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )
        
        return dataloader, input_fps

    def process_video(self, dataloader, input_fps):
        """Process video with GPU optimizations."""
        logger.info('Processing video with GPU optimizations')
        caption = "enhance video quality" + self.model.positive_prompt
        
        # Set up noise levels
        total_noise_levels = 900 if self.denoise_level > 0 else 600
        setup_seed(666)
        
        # Process video chunks
        output_chunks = []
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Process with AMP
                with autocast(enabled=self.amp):
                    output = self.model.test(
                        batch,
                        total_noise_levels,
                        steps=self.steps,
                        solver_mode=self.solver_mode,
                        guide_scale=self.guide_scale,
                        max_chunk_len=self.max_chunk_len,
                        denoise_level=self.denoise_level,
                        preserve_details=self.preserve_details
                    )
                
                output_chunks.append(output)
        
        # Combine chunks
        output = torch.cat(output_chunks, dim=0)
        output = tensor2vid(output)
        
        # Apply color fix
        output = adain_color_fix(output, video_data)
        
        # Save video
        save_video(
            output,
            os.path.dirname(self.output_path),
            os.path.basename(self.output_path),
            fps=input_fps
        )
        
        return self.output_path

class VideoDataset(torch.utils.data.Dataset):
    """Dataset for video processing with optimized memory usage."""
    def __init__(
        self,
        video_data: torch.Tensor,
        target_res: Tuple[int, int],
        max_chunk_len: int = 16,
        frame_stride: int = 8,
        resize_short_edge: int = 360
    ):
        self.video_data = video_data
        self.target_res = target_res
        self.max_chunk_len = max_chunk_len
        self.frame_stride = frame_stride
        self.resize_short_edge = resize_short_edge
        
        # Calculate number of chunks
        self.num_frames = video_data.shape[0]
        self.num_chunks = (self.num_frames - self.max_chunk_len) // self.frame_stride + 1
    
    def __len__(self):
        return self.num_chunks
    
    def __getitem__(self, idx):
        # Calculate frame indices for this chunk
        start_idx = idx * self.frame_stride
        end_idx = min(start_idx + self.max_chunk_len, self.num_frames)
        
        # Get chunk of frames
        chunk = self.video_data[start_idx:end_idx]
        
        # Resize if needed
        if self.resize_short_edge > 0:
            chunk = resize_video_chunk(chunk, self.resize_short_edge)
        
        return {
            'video_data': chunk,
            'target_res': self.target_res,
            'chunk_idx': idx
        }

def resize_video_chunk(chunk: torch.Tensor, short_edge: int) -> torch.Tensor:
    """Resize video chunk maintaining aspect ratio."""
    _, _, h, w = chunk.shape
    if h < w:
        new_h = short_edge
        new_w = int(w * (short_edge / h))
    else:
        new_w = short_edge
        new_h = int(h * (short_edge / w))
    
    return F.interpolate(
        chunk,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--input_path", required=True, type=str, help="input video path")
    parser.add_argument("--output_path", required=True, type=str, help="output video path")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--upscale", type=int, default=4, help='up-scale factor (2, 4, or 8)')
    parser.add_argument("--max_chunk_len", type=int, default=32, help='max chunk length for processing')
    parser.add_argument("--solver_mode", type=str, default='fast', help='fast | balanced | quality')
    parser.add_argument("--steps", type=int, default=15, help='number of steps')
    parser.add_argument("--denoise_level", type=float, default=0, help='denoise level (0-100)')
    parser.add_argument("--preserve_details", action='store_true', help='preserve video details')
    parser.add_argument("--frame_stride", type=int, default=8, help='frame stride for processing')
    parser.add_argument("--resize_short_edge", type=int, default=360, help='resize short edge for processing')
    parser.add_argument("--device", type=str, default='cuda', help='device to use (cuda or cpu)')
    parser.add_argument("--amp", action='store_true', help='use automatic mixed precision')
    parser.add_argument("--num_workers", type=int, default=8, help='number of workers for data loading')
    parser.add_argument("--pin_memory", action='store_true', help='pin memory for data loading')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size for processing')

    return parser.parse_args()

def main():
    args = parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path
    upscale = args.upscale
    max_chunk_len = args.max_chunk_len
    steps = args.steps
    solver_mode = args.solver_mode
    denoise_level = args.denoise_level
    preserve_details = args.preserve_details
    frame_stride = args.frame_stride
    resize_short_edge = args.resize_short_edge
    device = torch.device(args.device)
    amp = args.amp
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    batch_size = args.batch_size

    assert solver_mode in ('fast', 'balanced', 'quality')
    assert upscale in (2, 4, 8)
    assert 0 <= denoise_level <= 100

    guide_scale = 7.5  # Default value works well for most cases

    # Enable cuDNN benchmarking for better performance
    if device.type == 'cuda':
        cudnn.benchmark = True

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = VideoToVideo_sr(EasyDict(model_path=model_path))
    
    # Create data loader with optimized settings
    dataloader, input_fps = create_dataloader(
        input_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Process video with AMP if enabled
    with autocast(enabled=amp):
        process_video(
            model,
            dataloader,
            output_path,
            max_chunk_len,
            frame_stride,
            resize_short_edge,
            denoise_level,
            preserve_details,
            device
        )

def load_model(model_path, device):
    # Load model implementation here
    pass

def create_dataloader(input_path, batch_size, num_workers, pin_memory):
    # Create optimized data loader implementation here
    pass

def process_video(model, dataloader, output_path, max_chunk_len, frame_stride, resize_short_edge, denoise_level, preserve_details, device):
    # Video processing implementation here
    pass

if __name__ == '__main__':
    main()
