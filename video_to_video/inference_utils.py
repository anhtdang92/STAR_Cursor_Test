import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from typing import List, Dict, Any

def load_video(video_path: str) -> List[np.ndarray]:
    """Load video frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def save_video(frames: List[np.ndarray], output_path: str, fps: int = 30) -> str:
    """Save frames as a video file."""
    if not frames:
        raise ValueError("No frames to save")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return output_path

def preprocess_frames(frames: List[np.ndarray], device: str = 'cuda') -> torch.Tensor:
    """Preprocess video frames for model input."""
    # Convert frames to RGB
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    
    # Normalize and convert to tensor
    frames_tensor = torch.stack([
        torch.from_numpy(frame).float() / 255.0
        for frame in frames_rgb
    ])
    
    # Add batch dimension and move to device
    frames_tensor = frames_tensor.unsqueeze(0).to(device)
    
    return frames_tensor

def postprocess_frames(frames: torch.Tensor) -> List[np.ndarray]:
    """Convert model output frames back to video format."""
    # Move to CPU and convert to numpy
    frames = frames.cpu().numpy()
    
    # Remove batch dimension
    frames = frames.squeeze(0)
    
    # Denormalize and convert to BGR
    frames = [(frame * 255).astype(np.uint8) for frame in frames]
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    
    return frames

def process_video(
    model: torch.nn.Module,
    video_path: str,
    output_path: str,
    device: str = 'cuda',
    batch_size: int = 1
) -> str:
    """Process a video file using the model."""
    # Load video
    frames = load_video(video_path)
    
    # Process frames in batches
    processed_frames = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        
        # Preprocess batch
        batch_tensor = preprocess_frames(batch_frames, device)
        
        # Process batch
        with torch.no_grad():
            processed_batch = model.sample(batch_tensor)
        
        # Postprocess batch
        processed_batch_frames = postprocess_frames(processed_batch)
        processed_frames.extend(processed_batch_frames)
    
    # Save processed video
    output_path = save_video(processed_frames, output_path)
    
    return output_path

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get information about a video file."""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    
    cap.release()
    return info 