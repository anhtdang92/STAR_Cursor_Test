import unittest
from app import app
import os
import tempfile

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_upload_endpoint(self):
        # Test upload endpoint without file
        response = self.app.post('/api/upload')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No video file provided', response.data)

    def test_status_endpoint(self):
        # Test status endpoint with non-existent task
        response = self.app.get('/api/status/nonexistent')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Task not found', response.data)

    def test_download_endpoint(self):
        # Test download endpoint with non-existent task
        response = self.app.get('/api/download/nonexistent')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Task not found', response.data)

if __name__ == '__main__':
    unittest.main() 