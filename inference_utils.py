import os
import subprocess
import tempfile
import cv2
import torch
from PIL import Image
from typing import Mapping, Generator, List, Tuple
from einops import rearrange
import numpy as np
import torchvision.transforms.functional as transforms_F
from video_to_video.utils.logger import get_logger
from contextlib import contextmanager
import gc
import shutil

logger = get_logger()


def tensor2vid(video: torch.Tensor, mean: List[float] = [0.5, 0.5, 0.5], std: List[float] = [0.5, 0.5, 0.5]) -> np.ndarray:
    """Convert tensor to video with proper memory management"""
    try:
        # Create copies to avoid modifying input
        video = video.clone()
        mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
        std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
        
        # Process in chunks to manage memory
        chunk_size = 16  # Process 16 frames at a time
        total_frames = video.shape[2]
        processed_frames = []
        
        for i in range(0, total_frames, chunk_size):
            end_idx = min(i + chunk_size, total_frames)
            chunk = video[:, :, i:end_idx, :, :]
            
            # Process chunk
            chunk = chunk.mul(std).add(mean)
            chunk.clamp_(0, 1)
            chunk = chunk * 255.0
            
            # Convert to numpy and move to CPU
            chunk_np = chunk.cpu().numpy()
            processed_frames.append(chunk_np)
            
            # Clear GPU memory
            del chunk
            torch.cuda.empty_cache()
        
        # Combine processed chunks
        video_np = np.concatenate(processed_frames, axis=2)
        images = rearrange(video_np, 'b c f h w -> b f h w c')[0]
        
        return images
    except Exception as e:
        logger.error(f"Error in tensor2vid: {str(e)}")
        raise
    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()


def preprocess(input_frames: List[np.ndarray]) -> torch.Tensor:
    """Preprocess input frames with memory management"""
    try:
        out_frame_list = []
        for frame in input_frames:
            # Convert BGR to RGB
            frame = frame[:, :, ::-1]
            frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
            frame = transforms_F.to_tensor(frame)
            out_frame_list.append(frame)
        
        out_frames = torch.stack(out_frame_list, dim=0)
        out_frames.clamp_(0, 1)
        
        # Normalize
        mean = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
        std = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
        out_frames.sub_(mean.view(1, -1, 1, 1)).div_(std.view(1, -1, 1, 1))
        
        return out_frames
    except Exception as e:
        logger.error(f"Error in preprocess: {str(e)}")
        raise
    finally:
        # Cleanup
        del out_frame_list
        gc.collect()


def adjust_resolution(h: int, w: int, up_scale: float) -> Tuple[int, int]:
    """Adjust resolution with proper validation"""
    try:
        if h <= 0 or w <= 0 or up_scale <= 0:
            raise ValueError("Invalid input dimensions or scale factor")
            
        if h*up_scale < 720:
            up_s = 720/h
            target_h = int(up_s*h//2*2)
            target_w = int(up_s*w//2*2)
        elif h*w*up_scale*up_scale > 1280*2048:
            up_s = np.sqrt(1280*2048/(h*w))
            target_h = int(up_s*h//2*2)
            target_w = int(up_s*w//2*2)
        else:
            target_h = int(up_scale*h//2*2)
            target_w = int(up_scale*w//2*2)
            
        return (target_h, target_w)
    except Exception as e:
        logger.error(f"Error in adjust_resolution: {str(e)}")
        raise


@contextmanager
def video_capture(vid_path: str):
    """Context manager for video capture"""
    capture = None
    try:
        capture = cv2.VideoCapture(vid_path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {vid_path}")
        yield capture
    finally:
        if capture is not None:
            capture.release()


def load_video(vid_path: str) -> Tuple[List[np.ndarray], float]:
    """Load video with proper resource management"""
    try:
        with video_capture(vid_path) as capture:
            _fps = capture.get(cv2.CAP_PROP_FPS)
            _total_frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if _total_frame_num <= 0:
                raise ValueError(f"Invalid frame count in video: {vid_path}")
            
            frame_list = []
            stride = 1
            pointer = 0
            
            while len(frame_list) < _total_frame_num:
                ret, frame = capture.read()
                pointer += 1
                
                if not ret or frame is None:
                    break
                    
                if pointer >= _total_frame_num + 1:
                    break
                    
                if pointer % stride == 0:
                    frame_list.append(frame)
                    
                # Clear memory periodically
                if len(frame_list) % 100 == 0:
                    gc.collect()
            
            return frame_list, _fps
    except Exception as e:
        logger.error(f"Error in load_video: {str(e)}")
        raise


def save_video(video: List[np.ndarray], save_dir: str, file_name: str, fps: float = 16.0) -> None:
    """Save video with proper resource management and error handling"""
    temp_dir = None
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(save_dir, file_name)
        tmp_path = os.path.join(save_dir, 'tmp.mp4')
        
        # Save frames
        for fid, frame in enumerate(video):
            tpth = os.path.join(temp_dir, '%06d.png' % (fid + 1))
            cv2.imwrite(tpth, frame[:, :, ::-1])
        
        # Use subprocess.Popen with timeout
        cmd = [
            'ffmpeg', '-y',
            '-f', 'image2',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, '%06d.png'),
            '-vcodec', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '0',
            '-pix_fmt', 'yuv420p',
            tmp_path
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=300)  # 5-minute timeout
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
        except subprocess.TimeoutExpired:
            process.kill()
            raise TimeoutError("FFmpeg process timed out")
        
        # Move temporary file to final location
        os.rename(tmp_path, output_path)
        
    except Exception as e:
        logger.error(f"Error in save_video: {str(e)}")
        # Clean up temporary files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def make_mask_cond(in_f_num, interp_f_num):
    mask_cond = []
    interp_cond = [-1 for _ in range(interp_f_num)]
    for i in range(in_f_num):
        mask_cond.append(i)
        if i != in_f_num - 1:
            mask_cond += interp_cond
    return mask_cond


def collate_fn(data: Mapping, device: torch.device) -> Mapping:
    """Prepare input data with proper error handling"""
    try:
        from torch.utils.data.dataloader import default_collate
        
        if isinstance(data, (dict, Mapping)):
            return type(data)({
                k: collate_fn(v, device) if k != 'img_metas' else v
                for k, v in data.items()
            })
        elif isinstance(data, (tuple, list)):
            if len(data) == 0:
                return torch.Tensor([])
            if isinstance(data[0], (int, float)):
                return default_collate(data).to(device)
            else:
                return type(data)(collate_fn(v, device) for v in data)
        elif isinstance(data, np.ndarray):
            if data.dtype.type is np.str_:
                return data
            else:
                return collate_fn(torch.from_numpy(data), device)
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, (bytes, str, int, float, bool, type(None))):
            return data
        else:
            raise ValueError(f'Unsupported data type {type(data)}')
    except Exception as e:
        logger.error(f"Error in collate_fn: {str(e)}")
        raise