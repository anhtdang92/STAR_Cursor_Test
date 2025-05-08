import os
import uuid
import subprocess
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import time
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)
CORS(app)

# Task status tracking
tasks = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4'}

def process_video(task_id, input_path, output_path):
    try:
        # Update task status to processing
        tasks[task_id]['status'] = 'processing'
        
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
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            tasks[task_id]['status'] = 'complete'
            tasks[task_id]['download_url'] = f'/api/download/{task_id}'
        else:
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error'] = stderr.decode()
            
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
    finally:
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
        
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{task_id}_upscaled_{filename}")
    
    file.save(input_path)
    
    # Initialize task status
    tasks[task_id] = {
        'status': 'pending',
        'input_path': input_path,
        'output_path': output_path
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

@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
        
    return jsonify({
        'status': tasks[task_id]['status'],
        'downloadUrl': tasks[task_id].get('download_url'),
        'error': tasks[task_id].get('error')
    })

@app.route('/api/download/<task_id>', methods=['GET'])
def download_video(task_id):
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
        download_name=os.path.basename(task['output_path'])
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000) 