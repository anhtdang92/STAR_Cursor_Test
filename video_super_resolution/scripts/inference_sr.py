import os
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from argparse import ArgumentParser, Namespace
import json
from typing import Any, Dict, List, Mapping, Tuple
from easydict import EasyDict
from pathlib import Path
import cv2
import signal
from tqdm import tqdm
import torch.nn.functional as F
import traceback
import logging
from datetime import datetime
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(base_path)
from video_to_video.video_to_video_model import VideoToVideo_sr
from video_to_video.utils.seed import setup_seed
from video_to_video.utils.logger import get_logger
from video_super_resolution.color_fix import adain_color_fix

from inference_utils import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure logging to file with more detailed format
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'star_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# Add console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info(f"Logging initialized. Log file: {log_file}")
logger.info(f"Python version: {sys.version}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
logger.info(f"Current working directory: {os.getcwd()}")

# Set PyTorch thread optimization
logger.info("Setting PyTorch num_threads to 16")
torch.set_num_threads(16)

class GracefulInterruptHandler:
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        logger.info("GracefulInterruptHandler initialized.")

    def signal_handler(self, signum, frame):
        logger.info("Received interrupt signal. Will try to save partial progress...")
        self.interrupted = True

def save_partial_video(processed_frames, output_path, fps):
    """Save processed frames as a video, even if processing is incomplete."""
    if not processed_frames:
        logger.warning("No frames to save for partial video")
        return
    
    try:
        logger.info(f"Attempting to save partial video with {len(processed_frames)} frames to {output_path}")
        height, width = processed_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in processed_frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Saved partial video with {len(processed_frames)} frames to {output_path}")
    except Exception as e:
        logger.error(f"Error saving partial video: {str(e)}")

class VideoProcessor:
    def __init__(self, model, device):
        logger.info(f"VideoProcessor initializing with device: {device}")
        self.model = model
        self.device = device
        self.interrupt_handler = GracefulInterruptHandler()
        self.processed_frames = []
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0
        logger.info("VideoProcessor initialized.")

    def process_video(self, input_path, output_path, chunk_size=16):
        """Process video with progress tracking and partial saving support."""
        logger.info(f"VideoProcessor.process_video starting for input: {input_path}, output: {output_path}, chunk_size: {chunk_size}")
        try:
            # Get video info
            cap = cv2.VideoCapture(input_path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Video info: Total frames = {self.total_frames}, FPS = {self.fps}")
            
            # Progress bar setup
            pbar = tqdm(total=self.total_frames, desc="Processing frames")
            
            while cap.isOpened():
                frames_chunk = []
                for _ in range(chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames_chunk.append(frame)
                    self.current_frame += 1
                
                if not frames_chunk:
                    logger.info("No more frames in video to process.")
                    break
                
                logger.info(f"Processing chunk: current_frame = {self.current_frame}, frames in chunk = {len(frames_chunk)}")
                # Process chunk
                try:
                    processed_chunk = self.process_chunk(frames_chunk)
                    self.processed_frames.extend(processed_chunk)
                    pbar.update(len(frames_chunk))
                    logger.info(f"Chunk processed. Total processed_frames = {len(self.processed_frames)}")
                    
                    # Save progress periodically (every 100 frames)
                    if len(self.processed_frames) % 100 == 0:
                        logger.info(f"Reached {len(self.processed_frames)} processed frames, attempting periodic save.")
                        temp_output = output_path.replace('.mp4', '_partial.mp4')
                        save_partial_video(self.processed_frames, temp_output, self.fps)
                        
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    if self.processed_frames:
                        self.save_progress(output_path)
                    raise
                
                # Check for interruption
                if self.interrupt_handler.interrupted:
                    logger.info("Processing interrupted by signal, saving progress...")
                    self.save_progress(output_path)
                    break
            
            pbar.close()
            cap.release()
            logger.info("Video capture released.")
            
            # Save final video if not interrupted
            if not self.interrupt_handler.interrupted:
                logger.info("Processing complete, saving final video.")
                save_partial_video(self.processed_frames, output_path, self.fps)
            else:
                logger.info("Processing was interrupted, final save_partial_video call might have occurred in interruption block.")
                
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            if self.processed_frames:
                self.save_progress(output_path)
            raise
            
    def process_chunk(self, frames):
        """Process a chunk of frames using the model."""
        try:
            logger.info(f"VideoProcessor.process_chunk: processing {len(frames)} frames.")
            # Convert frames to tensor and process
            frames_tensor = self.prepare_frames(frames)
            logger.info(f"Frames tensor prepared, shape: {frames_tensor.shape}, type: {frames_tensor.dtype}, device: {frames_tensor.device}")
            with torch.no_grad():
                logger.info("Calling model for inference on chunk...")
                processed_tensor = self.model(frames_tensor)
                logger.info(f"Model inference complete. Processed tensor shape: {processed_tensor.shape}, type: {processed_tensor.dtype}, device: {processed_tensor.device}")
            
            # Convert back to numpy arrays
            processed_frames = self.tensor_to_frames(processed_tensor)
            logger.info(f"Processed tensor converted back to {len(processed_frames)} frames.")
            return processed_frames
            
        except Exception as e:
            logger.error(f"Error in process_chunk: {str(e)}")
            raise
            
    def save_progress(self, output_path):
        """Save current progress when interrupted."""
        if self.processed_frames:
            partial_output = output_path.replace('.mp4', '_partial.mp4')
            logger.info(f"VideoProcessor.save_progress: Attempting to save {len(self.processed_frames)} frames to {partial_output}")
            save_partial_video(self.processed_frames, partial_output, self.fps)
            logger.info(f"Saved {len(self.processed_frames)}/{self.total_frames} frames to partial file.")
        else:
            logger.warning("VideoProcessor.save_progress: No processed frames to save.")
            
    def get_progress(self):
        """Get current processing progress."""
        return {
            'current_frame': self.current_frame,
            'total_frames': self.total_frames,
            'percentage': (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0
        }

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
        logger.info("Initializing STAR model...")
        print("Initializing STAR model...", flush=True)
        logger.info(f"Output path: {output_path}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Device: {device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Model path: {model_path}", flush=True)
        print(f"Device: {device}", flush=True)
        
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

        logger.info(f"STAR model parameters:")
        logger.info(f"  solver_mode: {solver_mode}")
        logger.info(f"  steps: {steps}")
        logger.info(f"  guide_scale: {guide_scale}")
        logger.info(f"  upscale: {upscale}")
        logger.info(f"  max_chunk_len: {max_chunk_len}")
        logger.info(f"  frame_stride: {frame_stride}")
        logger.info(f"  resize_short_edge: {resize_short_edge}")
        logger.info(f"  denoise_level: {denoise_level}")
        logger.info(f"  preserve_details: {preserve_details}")
        logger.info(f"  amp: {amp}")
        logger.info(f"  num_workers: {num_workers}")
        logger.info(f"  pin_memory: {pin_memory}")
        logger.info(f"  batch_size: {batch_size}")

        # Enable cuDNN benchmarking for better performance
        if self.device.type == 'cuda':
            logger.info("Enabling cuDNN benchmarking.")
            cudnn.benchmark = True

        # Load model
        try:
            logger.info(f"Loading VideoToVideo_sr model from path: {model_path}")
            print(f"Loading VideoToVideo_sr model from path: {model_path}", flush=True)
            self.model = VideoToVideo_sr(EasyDict(model_path=model_path))
            logger.info("VideoToVideo_sr model loaded successfully.")
            print("VideoToVideo_sr model loaded successfully.", flush=True)
            
            logger.info(f"Moving model to device: {self.device}")
            print(f"Moving model to device: {self.device}", flush=True)
            self.model.to(self.device)
            logger.info(f"Model moved to device: {self.device}")
            print(f"Model moved to device: {self.device}", flush=True)
            
            logger.info("Setting model to eval mode.")
            self.model.eval()
            logger.info("Model initialization complete.")
            print("Model initialization complete.", flush=True)
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"Error initializing model: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

    def enhance_a_video(self, input_path: str):
        logger.info(f"STAR.enhance_a_video started for input: {input_path}")
        # Create data loader with optimized settings
        dataloader, input_fps, video_data = self.create_dataloader(input_path)
        
        # Process video with AMP if enabled
        logger.info(f"Processing video with AMP enabled: {self.amp}")
        with autocast(enabled=self.amp):
            self.process_video(dataloader, input_fps, video_data)
        logger.info(f"STAR.enhance_a_video finished for input: {input_path}")

    def create_dataloader(self, input_path: str):
        logger.info(f"STAR.create_dataloader: Loading video: {input_path}")
        print(f"STAR.create_dataloader: Loading video: {input_path}", flush=True)
        input_frames, input_fps = load_video(input_path)
        logger.info(f"STAR.create_dataloader: Input FPS: {input_fps}, Number of frames: {len(input_frames)}")
        print(f"STAR.create_dataloader: Input FPS: {input_fps}, Number of frames: {len(input_frames)}", flush=True)
        
        # Preprocess frames
        video_data = preprocess(input_frames)
        _, _, h, w = video_data.shape
        logger.info(f"STAR.create_dataloader: Input resolution: {(h, w)}")
        print(f"STAR.create_dataloader: Input resolution: {(h, w)}", flush=True)
        
        # Calculate target resolution
        target_h, target_w = h * self.upscale, w * self.upscale
        logger.info(f"STAR.create_dataloader: Target resolution: {(target_h, target_w)}")
        print(f"STAR.create_dataloader: Target resolution: {(target_h, target_w)}", flush=True)
        
        # Create dataset
        dataset = VideoDataset(
            video_data,
            target_res=(target_h, target_w),
            max_chunk_len=self.max_chunk_len,
            frame_stride=self.frame_stride,
            resize_short_edge=self.resize_short_edge
        )
        
        logger.info(f"STAR.create_dataloader: Dataset created. Max_chunk_len={self.max_chunk_len}, frame_stride={self.frame_stride}, resize_short_edge={self.resize_short_edge}")
        print(f"STAR.create_dataloader: Dataset created. Max_chunk_len={self.max_chunk_len}, frame_stride={self.frame_stride}, resize_short_edge={self.resize_short_edge}", flush=True)
        # Create data loader with optimized settings
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )
        
        logger.info(f"STAR.create_dataloader: DataLoader created with batch_size={self.batch_size}, num_workers={self.num_workers}, pin_memory={self.pin_memory}")
        print(f"STAR.create_dataloader: DataLoader created with batch_size={self.batch_size}, num_workers={self.num_workers}, pin_memory={self.pin_memory}", flush=True)
        return dataloader, input_fps, video_data

    def process_video(self, dataloader, input_fps, video_data):
        logger.info(f"STAR.process_video: Starting video processing. Input FPS: {input_fps}, Batches in dataloader: {len(dataloader)}")
        print(f"STAR.process_video: Starting video processing. Input FPS: {input_fps}, Batches in dataloader: {len(dataloader)}", flush=True)
        all_output_frames = []
        original_frames_for_color_fix = []
        logger.info("Preparing for frame processing loop.")
        print("Preparing for frame processing loop.", flush=True)
        pbar = tqdm(dataloader, desc="Processing video chunks")
        total_frames = video_data.shape[0]
        logger.info(f"Total frames to process: {total_frames}")
        print(f"Total frames to process: {total_frames}", flush=True)
        logger.info(f"Batch size: {self.batch_size}, Max chunk length: {self.max_chunk_len}, Frame stride: {self.frame_stride}")
        print(f"Batch size: {self.batch_size}, Max chunk length: {self.max_chunk_len}, Frame stride: {self.frame_stride}", flush=True)
        logger.info(f"Expected number of batches: {len(dataloader)}")
        print(f"Expected number of batches: {len(dataloader)}", flush=True)
        try:
            for i, batch in enumerate(pbar):
                logger.info(f"Processing batch {i+1}/{len(dataloader)}")
                print(f"Processing batch {i+1}/{len(dataloader)}", flush=True)
                try:
                    original_chunk = batch['original_video'].to(self.device)
                    video_chunk = batch['video_data'].to(self.device)
                    target_res_chunk = batch['target_res'] 
                    logger.info(f"  Original chunk shape: {original_chunk.shape}, Video chunk shape: {video_chunk.shape}")
                    print(f"  Original chunk shape: {original_chunk.shape}, Video chunk shape: {video_chunk.shape}", flush=True)
                    logger.info(f"  Target resolution: {target_res_chunk}")
                    print(f"  Target resolution: {target_res_chunk}", flush=True)
                    with torch.no_grad():
                        logger.info(f"  Calling model.test for batch {i+1}")
                        print(f"  Calling model.test for batch {i+1}", flush=True)
                        output_chunk = self.model.test({
                            'video_data': video_chunk,
                            'y': ['' for _ in range(video_chunk.shape[0])],  # Assuming empty prompts for now
                            'target_res': (target_res_chunk[0][0].item(), target_res_chunk[1][0].item())
                        }, steps=self.steps, guide_scale=self.guide_scale, max_chunk_len=self.max_chunk_len)
                        logger.info(f"  Model.test completed for batch {i+1}. Output chunk shape: {output_chunk.shape}")
                        print(f"  Model.test completed for batch {i+1}. Output chunk shape: {output_chunk.shape}", flush=True)
                    if self.preserve_details:
                        logger.info(f"  Applying Adain color fix for batch {i+1}")
                        print(f"  Applying Adain color fix for batch {i+1}", flush=True)
                        output_chunk = adain_color_fix(output_chunk, original_chunk)
                        logger.info(f"  Adain color fix applied for batch {i+1}")
                        print(f"  Adain color fix applied for batch {i+1}", flush=True)
                    output_chunk = output_chunk.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() # B C F H W -> F H W C
                    output_chunk = (output_chunk * 255).astype(np.uint8)
                    all_output_frames.extend([frame for frame in output_chunk])
                    original_frames_for_color_fix.extend([frame for frame in original_chunk.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()])
                    current_frame = min((i + 1) * self.batch_size * self.max_chunk_len, total_frames)
                    progress_msg = f"Processing frame {current_frame}/{total_frames}"
                    print(progress_msg, flush=True)
                    logger.info(progress_msg)
                    pbar.set_postfix_str(progress_msg)
                    if hasattr(self, 'output_path'):
                        task_id = extract_task_id(self.output_path)
                        logger.info(f"Updating backend progress for task {task_id}: {progress_msg}")
                        update_backend_progress(task_id, current_frame, total_frames, progress_msg)
                except Exception as e:
                    logger.error(f"Error processing batch {i+1}: {str(e)}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    print(f"Error processing batch {i+1}: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    raise
            logger.info(f"All batches processed. Total output frames collected: {len(all_output_frames)}")
            print(f"All batches processed. Total output frames collected: {len(all_output_frames)}", flush=True)
            if all_output_frames:
                logger.info(f"Saving processed video to {self.output_path} with FPS: {input_fps}")
                print(f"Saving processed video to {self.output_path} with FPS: {input_fps}", flush=True)
                save_video_frames(self.output_path, all_output_frames, input_fps)
                logger.info(f"Processed video saved successfully to {self.output_path}")
                print(f"Processed video saved successfully to {self.output_path}", flush=True)
            else:
                logger.warning("No output frames were generated to save.")
                print("No output frames were generated to save.", flush=True)
        except Exception as e:
            logger.error(f"Error in process_video: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"Error in process_video: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

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
        logger.info(f"VideoDataset initialized: num_frames={self.num_frames}, target_res={target_res}, max_chunk_len={max_chunk_len}, frame_stride={frame_stride}, resize_short_edge={resize_short_edge}")
    
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
    parser = ArgumentParser(description="Run video super-resolution inference.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the upscaled video.")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--upscale", type=int, default=4, help='up-scale factor (2, 4, or 8)')
    parser.add_argument("--max_chunk_len", type=int, default=32, help='max chunk length for processing')
    parser.add_argument("--solver_mode", type=str, default='fast', help='fast | balanced | quality')
    parser.add_argument("--steps", type=int, default=15, help='number of steps')
    parser.add_argument("--guide_scale", type=float, default=7.5, help='guidance scale for classifier-free guidance')
    parser.add_argument("--denoise_level", type=float, default=0, help='denoise level (0-100)')
    parser.add_argument("--preserve_details", action='store_true', help='preserve video details')
    parser.add_argument("--frame_stride", type=int, default=8, help='frame stride for processing')
    parser.add_argument("--resize_short_edge", type=int, default=360, help='resize short edge for processing')
    parser.add_argument("--device", type=str, default='cuda', help='device to use (cuda or cpu)')
    parser.add_argument("--amp", action='store_true', help='use automatic mixed precision')
    parser.add_argument("--num_workers", type=int, default=8, help='number of workers for data loading')
    parser.add_argument("--pin_memory", action='store_true', help='pin memory for data loading')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size for processing')

    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")
    return args

def main():
    logger.info("inference_sr.py main() function started.")
    logger.info(f"sys.argv: {sys.argv}")  # Debug: print command line arguments
    args = parse_args()
    logger.info(f"Parsed arguments: {args}")  # Debug: print parsed arguments

    setup_seed(42)
    logger.info("Seed set to 42.")

    # Create STAR model instance
    logger.info("Initializing STAR model...")
    try:
        STAR_model = STAR(
            output_path=args.output_path,
            model_path=args.model_path,
            solver_mode=args.solver_mode,
            steps=args.steps,
            guide_scale=args.guide_scale,
            upscale=args.upscale,
            max_chunk_len=args.max_chunk_len,
            frame_stride=args.frame_stride,
            resize_short_edge=args.resize_short_edge,
            denoise_level=args.denoise_level,
            preserve_details=args.preserve_details,
            device=args.device,
            amp=args.amp,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            batch_size=args.batch_size
        )
        logger.info("STAR model initialized successfully.")

        # Enhance the video
        logger.info(f"Starting video enhancement for: {args.input_path}")
        STAR_model.enhance_a_video(args.input_path)
        logger.info(f"Video enhancement finished for: {args.input_path}")
        logger.info("inference_sr.py main() function finished.")
    except Exception as e:
        logger.error(f"Error in main(): {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        print(f"Fatal error: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        print(f"Fatal error: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)

def load_model(model_path: str, device: str = 'cuda') -> VideoToVideo_sr:
    """Load the model from checkpoint."""
    # Load model configuration
    config = {
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
        'attn_type': 'vanilla'
    }
    
    # Initialize model
    model = VideoToVideo_sr(
        config,
        embed_dim=4,
        ckpt_path=model_path
    )
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model

def create_dataloader(input_path, batch_size, num_workers, pin_memory):
    # Create optimized data loader implementation here
    # This function seems to be a placeholder or an alternative not used by the STAR class.
    # The STAR class has its own create_dataloader method.
    logger.warning("Global create_dataloader function called, but STAR class uses its own method. This might be unused.")
    pass

def process_video(
    model: VideoToVideo_sr,
    input_path: str,
    output_path: str,
    device: str = 'cuda',
    batch_size: int = 1,
    frame_stride: int = 1,
    resize_short_edge: int = None,
    preserve_details: bool = True
) -> str:
    """
    Main function to process a video for super-resolution.
    This function also seems to be a placeholder or an alternative path.
    The primary path appears to be through the STAR class and main() function.
    """
    logger.warning("Global process_video function called. The primary path is usually through STAR class and main(). This might be an alternative or older path.")
    
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA is not available on this machine. Using CPU instead.")
        device = torch.device('cpu')

    logger.info(f"Starting video processing: {input_path} -> {output_path}")

    # Load model
    logger.info(f"Loading model from: {input_path}")
    model = load_model(input_path, str(device))
    logger.info("Model loaded successfully.")

    # Create video processor
    processor = VideoProcessor(model, device)
    logger.info("VideoProcessor created.")

    try:
        # Process video
        logger.info(f"Calling processor.process_video for {input_path}")
        processor.process_video(input_path, output_path)
        logger.info(f"Video processing finished for {input_path}. Output at {output_path}")

        # Optional: Apply color correction or other post-processing steps
        # ...

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        # Re-raise the exception to be caught by the backend
        raise
    finally:
        # Clean up resources if needed
        logger.info("process_video function completing.")
        # del model
        # torch.cuda.empty_cache()

    return output_path

def update_backend_progress(task_id, current_frame, total_frames, status=None):
    """Update the backend progress in tasks.json."""
    try:
        # Get the absolute path to the backend directory
        backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'backend')
        tasks_path = os.path.join(backend_dir, 'tasks.json')
        
        logger.info(f"Updating progress for task {task_id}: frame {current_frame}/{total_frames}")
        logger.info(f"Tasks file path: {tasks_path}")
        
        # Ensure the backend directory exists
        if not os.path.exists(backend_dir):
            logger.error(f"Backend directory not found: {backend_dir}")
            return
            
        # Read current tasks
        if os.path.exists(tasks_path):
            with open(tasks_path, 'r') as f:
                tasks = json.load(f)
        else:
            logger.warning(f"Tasks file not found, creating new one at {tasks_path}")
            tasks = {}
            
        # Update task progress
        if task_id in tasks:
            tasks[task_id]['current_frame'] = current_frame
            tasks[task_id]['total_frames'] = total_frames
            if status:
                tasks[task_id]['status'] = status
                
            # Calculate percentage
            if total_frames > 0:
                percentage = (current_frame / total_frames) * 100
                tasks[task_id]['progress'] = round(percentage, 2)
            
            # Write updated tasks
            with open(tasks_path, 'w') as f:
                json.dump(tasks, f, indent=2)
            logger.info(f"Progress updated successfully for task {task_id}")
        else:
            logger.warning(f"Task {task_id} not found in tasks.json")
            
    except Exception as e:
        logger.error(f"Failed to update backend progress: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

def extract_task_id(output_path):
    import re
    m = re.search(r'([a-f0-9\-]{36})_', os.path.basename(output_path))
    return m.group(1) if m else None
