import os
import uuid
import subprocess
import threading
import time
import json
import requests
import zipfile
import shutil
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, send_from_directory, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from config import Config
import sys
import traceback
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)
CORS(app)

# Task status tracking with more detailed information
tasks = {}

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'video_super_resolution', 'models', 'star.pth')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

print(f"Upload directory: {UPLOAD_FOLDER}")
print(f"Processed directory: {PROCESSED_FOLDER}")
print(f"Model path: {MODEL_PATH}")

def download_ffmpeg():
    """Download and setup FFmpeg for Windows"""
    try:
        # Create ffmpeg directory in the backend folder
        ffmpeg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg')
        os.makedirs(ffmpeg_dir, exist_ok=True)
        
        # Download FFmpeg
        logger.info("Downloading FFmpeg...")
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        zip_path = os.path.join(ffmpeg_dir, "ffmpeg.zip")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for data in response.iter_content(chunk_size=4096):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * downloaded / total_size)
                    logger.info(f"\rDownloading FFmpeg: [{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes")
        
        # Extract FFmpeg
        logger.info("\nExtracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
        
        # Move FFmpeg files to the correct location
        extracted_dir = os.path.join(ffmpeg_dir, "ffmpeg-master-latest-win64-gpl")
        bin_dir = os.path.join(ffmpeg_dir, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        
        for file in os.listdir(os.path.join(extracted_dir, "bin")):
            shutil.move(
                os.path.join(extracted_dir, "bin", file),
                os.path.join(bin_dir, file)
            )
        
        # Clean up
        shutil.rmtree(extracted_dir)
        os.remove(zip_path)
        
        # Add FFmpeg to PATH for this process
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
        
        logger.info("FFmpeg setup complete!")
        return True
    except Exception as e:
        logger.error(f"Error downloading FFmpeg: {str(e)}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("FFmpeg is installed and accessible")
            return True
        else:
            logger.error("FFmpeg is not accessible")
            return False
    except FileNotFoundError:
        logger.error("FFmpeg is not installed")
        return False

# Check FFmpeg on startup and download if needed
if not check_ffmpeg():
    logger.info("FFmpeg not found. Attempting to download...")
    if download_ffmpeg():
        if not check_ffmpeg():
            logger.error("Failed to setup FFmpeg. Please install it manually.")
    else:
        logger.error("""
        Failed to download FFmpeg automatically. Please install it manually:
        
        Windows:
        1. Download FFmpeg from https://ffmpeg.org/download.html
        2. Extract the files
        3. Add the bin folder to your system PATH
        
        Linux:
        sudo apt-get update && sudo apt-get install ffmpeg
        
        macOS:
        brew install ffmpeg
        """)

def cleanup_old_tasks():
    """Clean up tasks older than 24 hours"""
    current_time = datetime.now()
    for task_id in list(tasks.keys()):
        task = tasks[task_id]
        if current_time - task['created_at'] > timedelta(hours=24):
            try:
                if os.path.exists(task['input_path']):
                    os.remove(task['input_path'])
                if os.path.exists(task['output_path']):
                    os.remove(task['output_path'])
                del tasks[task_id]
            except Exception as e:
                logger.error(f"Error cleaning up task {task_id}: {str(e)}")

def allowed_file(filename):
    """Check if the file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4'}

def get_video_duration(input_path):
    """Get video duration using ffmpeg"""
    try:
        if not check_ffmpeg():
            raise Exception("FFmpeg is not installed or not accessible. Please install FFmpeg first.")
            
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-f', 'null',
            '-'
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Extract duration from ffmpeg output
        for line in result.stderr.split('\n'):
            if 'Duration' in line:
                time_str = line.split('Duration: ')[1].split(',')[0].strip()
                h, m, s = time_str.split(':')
                return float(h) * 3600 + float(m) * 60 + float(s)
                
        raise Exception("Could not find duration in FFmpeg output")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise Exception("Error running FFmpeg command")
    except Exception as e:
        logger.error(f"Error getting video duration: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise Exception(f"Error getting video duration: {str(e)}")

def get_star_args(settings):
    """Convert frontend settings to STAR model arguments"""
    try:
        # Validate scale factor
        scale_factor = int(settings.get('scaleFactor', 4))
        if scale_factor not in [2, 4, 8]:
            logger.warning(f"Invalid scale factor {scale_factor}, defaulting to 4")
            scale_factor = 4

        # Map model names to their corresponding paths
        model_map = {
            'Artemis': 'artemis.pth',
            'Gaia': 'gaia.pth',
            'Theia': 'theia.pth'
        }
        model_name = settings.get('model', 'Artemis')
        model_file = model_map.get(model_name, 'artemis.pth')

        # Get other settings
        denoise_level = float(settings.get('denoiseLevel', 0))
        preserve_details = settings.get('enhanceDetails', True)

        return [
            '--input_path', os.path.join(UPLOAD_FOLDER, 'input.mp4'),
            '--output_path', os.path.join(PROCESSED_FOLDER, 'output.mp4'),
            '--model_path', MODEL_PATH,
            '--upscale', str(scale_factor),
            '--denoise_level', str(denoise_level),
            '--device', 'cuda',
            '--amp',
            '--num_workers', '8',
            '--pin_memory',
            '--batch_size', '1',
            '--frame_length', '16',
            '--frame_stride', '8',
            '--resize_short_edge', '360'
        ] + (['--preserve_details'] if preserve_details else [])
    except Exception as e:
        logger.error(f"Error preparing STAR arguments: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

def process_video(task_id, input_path, output_path, settings):
    """Process video with progress tracking and error handling"""
    try:
        # Ensure upload and processed directories exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        
        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['progress'] = 0
        
        # Get video duration for progress calculation
        duration = get_video_duration(input_path)
        if duration is None:
            raise Exception("Could not determine video duration")
        
        # Get STAR model arguments from settings
        star_args = get_star_args(settings)
        
        # Construct the full command
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'video_super_resolution', 'scripts', 'inference_sr.py')
        
        cmd = [sys.executable, script_path] + star_args
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Input path exists: {os.path.exists(input_path)}")
        logger.info(f"Output directory exists: {os.path.exists(os.path.dirname(output_path))}")
        logger.info(f"Model path exists: {os.path.exists(app.config['MODEL_PATH'])}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # Monitor process and update progress
        while True:
            if process.poll() is not None:
                break
                
            # Update progress based on output file size
            if os.path.exists(output_path):
                current_size = os.path.getsize(output_path)
                expected_size = os.path.getsize(input_path) * settings.get('scale', 4)
                progress = min(95, (current_size / expected_size) * 100)
                tasks[task_id]['progress'] = progress
                
            time.sleep(1)
        
        stdout, stderr = process.communicate()
        
        # Log the complete output
        logger.info(f"Process stdout: {stdout}")
        if stderr:
            logger.error(f"Process stderr: {stderr}")
        
        if process.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                tasks[task_id]['status'] = 'complete'
                tasks[task_id]['progress'] = 100
                tasks[task_id]['download_url'] = f'/api/download/{task_id}'
                logger.info(f"Video processing completed successfully for task {task_id}")
            else:
                error_msg = "Output file not found or empty after processing"
                logger.error(error_msg)
                logger.error(f"Output path: {output_path}")
                logger.error(f"Output exists: {os.path.exists(output_path)}")
                if os.path.exists(output_path):
                    logger.error(f"Output size: {os.path.getsize(output_path)}")
                raise Exception(error_msg)
        else:
            error_msg = stderr if stderr else 'Unknown error during processing'
            logger.error(f"Processing failed for task {task_id}")
            logger.error(f"Return code: {process.returncode}")
            logger.error(f"Error message: {error_msg}")
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"Error processing video for task {task_id}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
    finally:
        # Clean up input file
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception as e:
            logger.error(f"Error cleaning up input file: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")

@app.route('/')
def index():
    return redirect('/test')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload with validation and error handling"""
    try:
        # Clean up old tasks
        cleanup_old_tasks()
        
        # Ensure upload and processed directories exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
            
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only MP4 files are allowed'}), 400
            
        # Check file size
        if request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': f"File too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB"}), 400
            
        # Parse settings
        settings = {}
        if 'settings' in request.form:
            try:
                settings = json.loads(request.form['settings'])
            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid settings format'}), 400
            
        # Generate unique task ID and save file
        task_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{task_id}_upscaled_{filename}")
        
        file.save(input_path)
        
        # Initialize task status
        tasks[task_id] = {
            'status': 'pending',
            'progress': 0,
            'input_path': input_path,
            'output_path': output_path,
            'created_at': datetime.now(),
            'filename': filename,
            'settings': settings
        }
        
        # Start processing in background
        thread = threading.Thread(
            target=process_video,
            args=(task_id, input_path, output_path, settings)
        )
        thread.daemon = True  # Make thread daemon so it doesn't block application shutdown
        thread.start()
        
        return jsonify({
            'taskId': task_id,
            'message': 'Video upload successful, processing started'
        })
        
    except Exception as e:
        logger.error(f"Error handling upload: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Server error processing upload'}), 500

@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Get task status with detailed information"""
    try:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
            
        task = tasks[task_id]
        return jsonify({
            'status': task['status'],
            'progress': task['progress'],
            'downloadUrl': task.get('download_url'),
            'error': task.get('error'),
            'filename': task['filename'],
            'settings': task.get('settings', {})
        })
        
    except Exception as e:
        logger.error(f"Error getting status for task {task_id}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Server error getting status'}), 500

@app.route('/api/download/<task_id>', methods=['GET'])
def download_video(task_id):
    """Download processed video with error handling"""
    try:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
            
        task = tasks[task_id]
        if task['status'] != 'complete':
            return jsonify({'error': 'Video processing not complete'}), 400
            
        if not os.path.exists(task['output_path']):
            return jsonify({'error': 'Processed video not found'}), 404
            
        return send_file(
            task['output_path'],
            as_attachment=True,
            download_name=f"upscaled_{task['filename']}"
        )
        
    except Exception as e:
        logger.error(f"Error downloading video for task {task_id}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Server error downloading video'}), 500

# Add a test route
@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({
        'status': 'success',
        'message': 'API is working!',
        'endpoints': {
            'upload': '/api/upload (POST)',
            'status': '/api/status/<task_id> (GET)',
            'download': '/api/download/<task_id> (GET)'
        }
    })

# Add error handler for method not allowed
@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        'error': 'Method not allowed',
        'message': f'The {request.method} method is not allowed for the requested URL.',
        'allowed_methods': e.valid_methods
    }), 405

# Add route to serve test page
@app.route('/test')
def test_page():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'test.html')

if __name__ == '__main__':
    # Check FFmpeg installation
    if not check_ffmpeg():
        logger.error("FFmpeg is not installed. Please install FFmpeg to continue.")
        sys.exit(1)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Using CPU instead.")
    
    app.run(debug=True, port=5000) 