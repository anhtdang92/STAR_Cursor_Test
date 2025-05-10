import pytest
import os
import torch
import numpy as np
from pathlib import Path
import sys
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import setup_logger, log_error, log_cuda_info, log_model_info

# Setup test logger
test_logger = setup_logger('star_diffusion.tests', 'backend/logs/test.log')

class TestEnvironment:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.test_data_dir = Path('backend/tests/test_data')
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir = Path('backend/tests/test_output')
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log test environment setup
        test_logger.info("Setting up test environment")
        log_cuda_info()
        
        yield
        
        # Cleanup after tests
        test_logger.info("Cleaning up test environment")

class TestCUDA(TestEnvironment):
    def test_cuda_availability(self):
        """Test CUDA availability and device properties"""
        assert torch.cuda.is_available(), "CUDA is not available"
        test_logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        
    def test_gpu_memory(self):
        """Test GPU memory allocation"""
        if torch.cuda.is_available():
            # Allocate a tensor on GPU
            tensor = torch.randn(1000, 1000).cuda()
            memory_allocated = torch.cuda.memory_allocated(0)
            test_logger.info(f"Memory allocated: {memory_allocated / 1024**2:.2f} MB")
            assert memory_allocated > 0, "Failed to allocate memory on GPU"
            
            # Clear memory
            del tensor
            torch.cuda.empty_cache()

class TestModel(TestEnvironment):
    def test_model_loading(self):
        """Test model loading and basic properties"""
        try:
            # Import your model here
            from inference_utils import load_model  # Adjust import as needed
            
            model = load_model()
            assert model is not None, "Failed to load model"
            log_model_info(model)
            
            # Test model forward pass
            test_input = torch.randn(1, 3, 256, 256).cuda()  # Adjust dimensions as needed
            with torch.no_grad():
                output = model(test_input)
            assert output is not None, "Model forward pass failed"
            
        except Exception as e:
            log_error(e, "Model loading test failed")
            raise

class TestVideoProcessing(TestEnvironment):
    def test_video_upload(self):
        """Test video upload functionality"""
        test_video_path = self.test_data_dir / "test_video.mp4"
        assert test_video_path.exists(), "Test video file not found"
        
    def test_video_processing(self):
        """Test video processing pipeline"""
        try:
            # Import your video processing functions here
            from inference_utils import process_video  # Adjust import as needed
            
            input_path = self.test_data_dir / "test_video.mp4"
            output_path = self.test_output_dir / "processed_video.mp4"
            
            # Process video
            result = process_video(str(input_path), str(output_path))
            assert result is not None, "Video processing failed"
            assert output_path.exists(), "Processed video file not created"
            
        except Exception as e:
            log_error(e, "Video processing test failed")
            raise

class TestAPI(TestEnvironment):
    def test_api_endpoints(self):
        """Test API endpoints"""
        try:
            import requests
            base_url = "http://localhost:5000"  # Adjust as needed
            
            # Test health check endpoint
            response = requests.get(f"{base_url}/health")
            assert response.status_code == 200, "Health check failed"
            
            # Test video upload endpoint
            test_video_path = self.test_data_dir / "test_video.mp4"
            with open(test_video_path, 'rb') as f:
                files = {'video': f}
                response = requests.post(f"{base_url}/upload", files=files)
            assert response.status_code == 200, "Video upload failed"
            
        except Exception as e:
            log_error(e, "API test failed")
            raise

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 