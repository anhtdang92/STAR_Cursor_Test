import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from typing import List, Dict, Any
from debug_utils import debug_cuda_memory, debug_system_resources, timing_decorator
from video_to_video.utils.logger import get_logger

logger = get_logger()

@timing_decorator
def load_video(video_path: str) -> List[np.ndarray]:
    """Load video frames from a video file with timing and debugging."""
    logger.info(f"Loading video from {video_path}")
    debug_system_resources()
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    
    logger.info(f"Loaded {len(frames)} frames")
    return frames

@timing_decorator
def save_video(frames: List[np.ndarray], output_path: str, fps: int = 30) -> str:
    """Save frames as a video file with timing and debugging."""
    logger.info(f"Saving video to {output_path}")
    debug_system_resources()
    
    if not frames:
        raise ValueError("No frames to save")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        for frame in frames:
            out.write(frame)
    finally:
        out.release()
    
    logger.info(f"Saved video with {len(frames)} frames at {fps} FPS")
    return output_path

@timing_decorator
def preprocess_frames(frames: List[np.ndarray], device: str = 'cuda') -> torch.Tensor:
    """Preprocess video frames for model input with timing and debugging."""
    logger.info("Preprocessing frames")
    debug_cuda_memory()
    debug_system_resources()
    
    # Convert frames to RGB
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    
    # Normalize and convert to tensor
    frames_tensor = torch.stack([
        torch.from_numpy(frame).float() / 255.0
        for frame in frames_rgb
    ])
    
    # Add batch dimension and move to device
    frames_tensor = frames_tensor.unsqueeze(0).to(device)
    
    logger.info(f"Preprocessed frames shape: {frames_tensor.shape}")
    return frames_tensor

@timing_decorator
def postprocess_frames(frames: torch.Tensor) -> List[np.ndarray]:
    """Convert model output frames back to video format with timing and debugging."""
    logger.info("Postprocessing frames")
    debug_cuda_memory()
    debug_system_resources()
    
    # Move to CPU and convert to numpy
    frames = frames.cpu().numpy()
    
    # Remove batch dimension
    frames = frames.squeeze(0)
    
    # Denormalize and convert to BGR
    frames = [(frame * 255).astype(np.uint8) for frame in frames]
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    
    logger.info(f"Postprocessed {len(frames)} frames")
    return frames

@timing_decorator
def process_video(
    model: torch.nn.Module,
    video_path: str,
    output_path: str,
    device: str = 'cuda',
    batch_size: int = 1
) -> str:
    """Process a video file using the model with timing and debugging."""
    logger.info(f"Processing video: {video_path}")
    debug_cuda_memory()
    debug_system_resources()
    
    # Load video
    frames = load_video(video_path)
    
    # Process frames in batches
    processed_frames = []
    for i in tqdm(range(0, len(frames), batch_size), desc="Processing frames"):
        batch_frames = frames[i:i + batch_size]
        
        # Preprocess batch
        batch_tensor = preprocess_frames(batch_frames, device)
        
        # Process batch
        with torch.no_grad():
            processed_batch = model.sample(batch_tensor)
        
        # Postprocess batch
        processed_batch_frames = postprocess_frames(processed_batch)
        processed_frames.extend(processed_batch_frames)
        
        # Log memory usage after each batch
        debug_cuda_memory()
    
    # Save processed video
    output_path = save_video(processed_frames, output_path)
    
    logger.info(f"Video processing complete. Output saved to: {output_path}")
    return output_path

@timing_decorator
def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get information about a video file with timing and debugging."""
    logger.info(f"Getting video info for: {video_path}")
    debug_system_resources()
    
    cap = cv2.VideoCapture(video_path)
    
    try:
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }
        
        logger.info(f"Video info: {info}")
        return info
    finally:
        cap.release() 