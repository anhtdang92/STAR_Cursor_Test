import unittest
import os
import json
import shutil
from datetime import datetime, timedelta
from io import BytesIO
from app import app, tasks, cleanup_old_tasks

class TestApp(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()
        
        # Create test directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        
        # Clear tasks
        tasks.clear()

    def tearDown(self):
        # Clean up test directories
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        shutil.rmtree(app.config['PROCESSED_FOLDER'])
        
        # Clear tasks
        tasks.clear()

    def test_upload_no_file(self):
        response = self.app.post('/api/upload')
        self.assertEqual(response.status_code, 400)
        self.assertIn('No video file provided', response.get_json()['error'])

    def test_upload_empty_file(self):
        response = self.app.post('/api/upload', data={
            'video': (BytesIO(), '')
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('No selected file', response.get_json()['error'])

    def test_upload_invalid_file_type(self):
        response = self.app.post('/api/upload', data={
            'video': (BytesIO(b'test data'), 'test.txt')
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid file type', response.get_json()['error'])

    def test_upload_file_too_large(self):
        # Create a file larger than MAX_CONTENT_LENGTH
        large_data = b'0' * (app.config['MAX_CONTENT_LENGTH'] + 1)
        response = self.app.post('/api/upload', data={
            'video': (BytesIO(large_data), 'test.mp4')
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('File too large', response.get_json()['error'])

    def test_upload_success(self):
        response = self.app.post('/api/upload', data={
            'video': (BytesIO(b'test video data'), 'test.mp4')
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('taskId', data)
        self.assertIn('message', data)
        
        # Verify task was created
        task_id = data['taskId']
        self.assertIn(task_id, tasks)
        self.assertEqual(tasks[task_id]['status'], 'pending')
        self.assertEqual(tasks[task_id]['progress'], 0)

    def test_status_not_found(self):
        response = self.app.get('/api/status/nonexistent')
        self.assertEqual(response.status_code, 404)
        self.assertIn('Task not found', response.get_json()['error'])

    def test_status_success(self):
        # Create a test task
        task_id = 'test-task'
        tasks[task_id] = {
            'status': 'processing',
            'progress': 50,
            'filename': 'test.mp4',
            'created_at': datetime.now()
        }
        
        response = self.app.get(f'/api/status/{task_id}')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'processing')
        self.assertEqual(data['progress'], 50)
        self.assertEqual(data['filename'], 'test.mp4')

    def test_download_not_found(self):
        response = self.app.get('/api/download/nonexistent')
        self.assertEqual(response.status_code, 404)
        self.assertIn('Task not found', response.get_json()['error'])

    def test_download_not_complete(self):
        # Create a test task that's still processing
        task_id = 'test-task'
        tasks[task_id] = {
            'status': 'processing',
            'progress': 50,
            'filename': 'test.mp4',
            'created_at': datetime.now()
        }
        
        response = self.app.get(f'/api/download/{task_id}')
        self.assertEqual(response.status_code, 400)
        self.assertIn('Video processing not complete', response.get_json()['error'])

    def test_cleanup_old_tasks(self):
        # Create an old task
        old_task_id = 'old-task'
        old_input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'old_test.mp4')
        old_output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'old_test_upscaled.mp4')
        
        # Create test files
        with open(old_input_path, 'wb') as f:
            f.write(b'test data')
        with open(old_output_path, 'wb') as f:
            f.write(b'test data')
        
        tasks[old_task_id] = {
            'status': 'complete',
            'input_path': old_input_path,
            'output_path': old_output_path,
            'created_at': datetime.now() - timedelta(hours=25)
        }
        
        # Create a recent task
        recent_task_id = 'recent-task'
        tasks[recent_task_id] = {
            'status': 'processing',
            'created_at': datetime.now()
        }
        
        # Run cleanup
        cleanup_old_tasks()
        
        # Verify old task was cleaned up
        self.assertNotIn(old_task_id, tasks)
        self.assertFalse(os.path.exists(old_input_path))
        self.assertFalse(os.path.exists(old_output_path))
        
        # Verify recent task remains
        self.assertIn(recent_task_id, tasks)

if __name__ == '__main__':
    unittest.main() 