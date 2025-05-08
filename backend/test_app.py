import unittest
from app import app
import os
import tempfile
import json
from werkzeug.datastructures import FileStorage

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.upload_dir = os.path.join(self.test_dir, 'input', 'video')
        self.output_dir = os.path.join(self.test_dir, 'output', 'video')
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.test_dir)

    def test_upload_endpoint_no_file(self):
        # Test upload endpoint without file
        response = self.app.post('/api/upload')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No video file provided')

    def test_upload_endpoint_invalid_file(self):
        # Test upload endpoint with invalid file type
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            temp_file.write(b'This is not a video file')
            temp_file.seek(0)
            response = self.app.post('/api/upload', data={
                'file': (temp_file, 'test.txt')
            }, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertEqual(data['error'], 'Invalid file type')

    def test_upload_endpoint_valid_file(self):
        # Test upload endpoint with valid MP4 file
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            temp_file.write(b'fake video content')
            temp_file.seek(0)
            response = self.app.post('/api/upload', data={
                'file': (temp_file, 'test.mp4')
            }, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 202)
            data = json.loads(response.data)
            self.assertIn('id', data)
            self.assertIn('status', data)
            self.assertEqual(data['status'], 'processing')

    def test_status_endpoint_nonexistent(self):
        # Test status endpoint with non-existent task
        response = self.app.get('/api/status/nonexistent')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Task not found')

    def test_status_endpoint_valid(self):
        # Test status endpoint with valid task
        # First create a task
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            temp_file.write(b'fake video content')
            temp_file.seek(0)
            response = self.app.post('/api/upload', data={
                'file': (temp_file, 'test.mp4')
            }, content_type='multipart/form-data')
            data = json.loads(response.data)
            task_id = data['id']

            # Then check its status
            response = self.app.get(f'/api/status/{task_id}')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('status', data)

    def test_download_endpoint_nonexistent(self):
        # Test download endpoint with non-existent task
        response = self.app.get('/api/download/nonexistent')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Task not found')

    def test_download_endpoint_not_ready(self):
        # Test download endpoint with task not ready
        # First create a task
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            temp_file.write(b'fake video content')
            temp_file.seek(0)
            response = self.app.post('/api/upload', data={
                'file': (temp_file, 'test.mp4')
            }, content_type='multipart/form-data')
            data = json.loads(response.data)
            task_id = data['id']

            # Then try to download before it's ready
            response = self.app.get(f'/api/download/{task_id}')
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertEqual(data['error'], 'Video processing not complete')

if __name__ == '__main__':
    unittest.main() 