import os
import uuid
import subprocess
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)
CORS(app)

# Task status tracking with more detailed information
tasks = {}

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
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-f', 'null',
            '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.PIPE)
        
        # Extract duration from ffmpeg output
        for line in result.stderr.split('\n'):
            if 'Duration' in line:
                time_str = line.split('Duration: ')[1].split(',')[0].strip()
                h, m, s = time_str.split(':')
                return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception as e:
        logger.error(f"Error getting video duration: {str(e)}")
        return None

def process_video(task_id, input_path, output_path):
    """Process video with progress tracking and error handling"""
    try:
        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['progress'] = 0
        
        # Get video duration for progress calculation
        duration = get_video_duration(input_path)
        if duration is None:
            raise Exception("Could not determine video duration")
        
        # Run STAR inference
        cmd = [
            'bash', 'video_super_resolution/scripts/inference_sr.sh',
            '--input_video', input_path,
            '--output_video', output_path,
            '--model_path', app.config['MODEL_PATH']
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # Monitor process and update progress
        while True:
            if process.poll() is not None:
                break
                
            # Update progress based on output file size
            if os.path.exists(output_path):
                current_size = os.path.getsize(output_path)
                expected_size = os.path.getsize(input_path) * 4  # Assuming 4x upscaling
                progress = min(95, (current_size / expected_size) * 100)
                tasks[task_id]['progress'] = progress
                
            time.sleep(1)
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            tasks[task_id]['status'] = 'complete'
            tasks[task_id]['progress'] = 100
            tasks[task_id]['download_url'] = f'/api/download/{task_id}'
        else:
            error_msg = stderr.decode() if stderr else 'Unknown error during processing'
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"Error processing video for task {task_id}: {str(e)}")
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
    finally:
        # Clean up input file
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception as e:
            logger.error(f"Error cleaning up input file: {str(e)}")

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload with validation and error handling"""
    try:
        # Clean up old tasks
        cleanup_old_tasks()
        
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
            'filename': filename
        }
        
        # Start processing in background
        thread = threading.Thread(
            target=process_video,
            args=(task_id, input_path, output_path)
        )
        thread.start()
        
        return jsonify({
            'taskId': task_id,
            'message': 'Video upload successful, processing started'
        })
        
    except Exception as e:
        logger.error(f"Error handling upload: {str(e)}")
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
            'filename': task['filename']
        })
        
    except Exception as e:
        logger.error(f"Error getting status for task {task_id}: {str(e)}")
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
        return jsonify({'error': 'Server error downloading video'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 