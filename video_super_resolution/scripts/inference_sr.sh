#!/bin/bash

# Set CPU thread optimization
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Default values
input_video=""
output_video=""
model_path=""
scale_factor=4
quality="balanced"
denoise_level=0
preserve_details=true
frame_length=16
frame_stride=8
resize_short_edge=360
device="cuda"

# Get the Python path from the virtual environment
PYTHON_PATH="$(which python)"
if [ -z "$PYTHON_PATH" ]; then
    # Try to find Python in the virtual environment
    if [ -d "../../.venv" ]; then
        if [ -f "../../.venv/Scripts/python" ]; then
            PYTHON_PATH="../../.venv/Scripts/python"
        elif [ -f "../../.venv/bin/python" ]; then
            PYTHON_PATH="../../.venv/bin/python"
        fi
    fi
fi

if [ -z "$PYTHON_PATH" ]; then
    echo "Error: Python not found. Please make sure Python is installed and in your PATH"
    exit 1
fi

echo "Using Python from: $PYTHON_PATH"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_video)
            input_video="$2"
            shift 2
            ;;
        --output_video)
            output_video="$2"
            shift 2
            ;;
        --model_path)
            model_path="$2"
            shift 2
            ;;
        --scale_factor)
            scale_factor="$2"
            shift 2
            ;;
        --quality)
            quality="$2"
            shift 2
            ;;
        --denoise_level)
            denoise_level="$2"
            shift 2
            ;;
        --preserve_details)
            preserve_details=true
            shift
            ;;
        --frame_length)
            frame_length="$2"
            shift 2
            ;;
        --frame_stride)
            frame_stride="$2"
            shift 2
            ;;
        --resize_short_edge)
            resize_short_edge="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$input_video" ] || [ -z "$output_video" ] || [ -z "$model_path" ]; then
    echo "Missing required parameters. Usage:"
    echo "inference_sr.sh --input_video <path> --output_video <path> --model_path <path>"
    exit 1
fi

echo "Processing video: $input_video"
echo "Output path: $output_video"
echo "Model path: $model_path"
echo "Scale factor: $scale_factor"
echo "Quality: $quality"
echo "Frame length: $frame_length"
echo "Frame stride: $frame_stride"
echo "Resize short edge: $resize_short_edge"
echo "Device: $device"
    
# Set solver mode based on quality
solver_mode="balanced"
steps=20
if [ "$quality" = "fast" ]; then
    solver_mode="fast"
    steps=15
elif [ "$quality" = "quality" ]; then
    solver_mode="quality"
    steps=25
fi
        
# Run Python script with optimized parameters
"$PYTHON_PATH" \
        ./video_super_resolution/scripts/inference_sr.py \
    --solver_mode "$solver_mode" \
    --steps "$steps" \
    --input_path "$input_video" \
    --output_path "$output_video" \
    --model_path "$model_path" \
    --upscale "$scale_factor" \
    --max_chunk_len "$frame_length" \
    --frame_stride "$frame_stride" \
    --resize_short_edge "$resize_short_edge" \
    --device "$device" \
    --denoise_level "$denoise_level" \
    --amp \
    --num_workers 8 \
    --pin_memory \
    --batch_size 1 \
    $([ "$preserve_details" = true ] && echo "--preserve_details")

if [ $? -eq 0 ]; then
    echo "Video processed successfully"
    exit 0
else
    echo "Error processing video"
    exit 1
fi
