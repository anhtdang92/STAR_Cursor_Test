# Video Super-Resolution Backend

A robust backend service for video super-resolution processing using the STAR model. This service provides a RESTful API for uploading, processing, and downloading upscaled videos with various quality settings.

## Features

- Video upload and processing with progress tracking
- Multiple quality presets (fast, balanced, quality)
- Configurable upscaling factors (2x, 4x, 8x)
- Denoising and detail preservation options
- Real-time progress monitoring
- Automatic cleanup of processed files
- Comprehensive error handling and logging
- GPU acceleration support
- Memory-efficient processing

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg
- PyTorch
- Flask
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd video-super-resolution
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the STAR model:
```bash
# Place the model file in video_super_resolution/models/light_deg.pt
# Place the config file in video_super_resolution/models/config.json
```

## Configuration

The service can be configured through environment variables or the `config.py` file:

- `SECRET_KEY`: Flask secret key
- `MAX_CONTENT_LENGTH`: Maximum upload file size (default: 100MB)
- `UPLOAD_FOLDER`: Directory for uploaded videos
- `PROCESSED_FOLDER`: Directory for processed videos
- `MODEL_PATH`: Path to the STAR model file
- `MODEL_CONFIG_PATH`: Path to the model configuration file

## Usage

1. Start the backend service:
```bash
python backend/app.py
```

2. The API will be available at `http://localhost:5000`

### API Endpoints

- `POST /api/upload`: Upload a video for processing
  - Required fields: video file, settings (JSON)
  - Returns: task ID

- `GET /api/status/<task_id>`: Get processing status
  - Returns: status, progress, download URL

- `GET /api/download/<task_id>`: Download processed video
  - Returns: processed video file

- `GET /api/progress/<task_id>`: Get detailed progress
  - Returns: current frame, total frames, progress percentage

- `POST /api/cancel/<task_id>`: Cancel processing
  - Returns: confirmation message

### Example Request

```python
import requests

# Upload video
files = {'video': open('input.mp4', 'rb')}
settings = {
    'scale': 4,
    'denoiseLevel': 0,
    'preserveDetails': True,
    'quality': 'balanced',
    'guideScale': 7.5
}
data = {'settings': json.dumps(settings)}

response = requests.post('http://localhost:5000/api/upload', 
                        files=files, 
                        data=data)
task_id = response.json()['taskId']

# Check status
status = requests.get(f'http://localhost:5000/api/status/{task_id}').json()
```

## Project Structure

```
video-super-resolution/
├── backend/
│   ├── app.py              # Main Flask application
│   ├── config.py           # Configuration settings
│   ├── debug_utils.py      # Debugging utilities
│   ├── logger.py           # Logging configuration
│   └── inference_utils.py  # Video processing utilities
├── video_super_resolution/
│   ├── models/             # Model files
│   └── scripts/            # Processing scripts
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Error Handling

The service includes comprehensive error handling for:
- File upload errors
- Processing errors
- Resource management
- Memory management
- GPU errors
- Network errors

All errors are logged with detailed information for debugging.

## Memory Management

The service implements several memory optimization techniques:
- Chunked video processing
- Automatic GPU memory cleanup
- Temporary file management
- Resource cleanup on errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]

## Acknowledgments

- STAR model authors
- FFmpeg project
- PyTorch team
