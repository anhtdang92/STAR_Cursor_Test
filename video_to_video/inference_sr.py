import os
import torch
import argparse
from modules.autoencoder import AutoencoderKLTemporalDecoder
from modules.video_to_video_model import VideoToVideo_sr
from inference_utils import process_video, get_video_info
from debug_utils import debug_cuda_memory, debug_system_resources, timing_decorator
from video_to_video.utils.logger import get_logger

logger = get_logger()

@timing_decorator
def load_model(model_path, device='cuda'):
    """Load the model from checkpoint with timing and debugging."""
    logger.info("Loading model...")
    debug_cuda_memory()
    debug_system_resources()
    
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
    
    logger.info("Model loaded successfully")
    debug_cuda_memory()
    return model

def main():
    parser = argparse.ArgumentParser(description='Video Super Resolution')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debugging')
    args = parser.parse_args()
    
    try:
        # Check if input file exists
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input video not found: {args.input}")
        
        # Check if model file exists
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
        
        # Get video information
        video_info = get_video_info(args.input)
        logger.info(f"Processing video: {args.input}")
        logger.info(f"Resolution: {video_info['width']}x{video_info['height']}")
        logger.info(f"FPS: {video_info['fps']}")
        logger.info(f"Duration: {video_info['duration']} seconds")
        logger.info(f"Total frames: {video_info['frame_count']}")
        
        # Log system and CUDA information
        debug_system_resources()
        debug_cuda_memory()
        
        # Load model
        logger.info("Loading model...")
        model = load_model(args.model, args.device)
        
        # Process video
        logger.info("Processing video...")
        output_path = process_video(
            model,
            args.input,
            args.output,
            device=args.device,
            batch_size=args.batch_size
        )
        
        logger.info(f"Video processing complete. Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}", exc_info=True)
        raise
    finally:
        # Log final system state
        debug_system_resources()
        debug_cuda_memory()

if __name__ == "__main__":
    main() 