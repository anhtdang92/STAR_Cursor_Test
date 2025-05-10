import os
import torch
import argparse
from modules.autoencoder import AutoencoderKLTemporalDecoder
from modules.video_to_video_model import VideoToVideo_sr
from inference_utils import process_video, get_video_info

def load_model(model_path, device='cuda'):
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

def main():
    parser = argparse.ArgumentParser(description='Video Super Resolution')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    
    # Check if model file exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    
    # Get video information
    video_info = get_video_info(args.input)
    print(f"Processing video: {args.input}")
    print(f"Resolution: {video_info['width']}x{video_info['height']}")
    print(f"FPS: {video_info['fps']}")
    print(f"Duration: {video_info['duration']} seconds")
    print(f"Total frames: {video_info['frame_count']}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model, args.device)
    
    # Process video
    print("Processing video...")
    output_path = process_video(
        model,
        args.input,
        args.output,
        device=args.device,
        batch_size=args.batch_size
    )
    
    print(f"Video processing complete. Output saved to: {output_path}")

if __name__ == '__main__':
    main() 