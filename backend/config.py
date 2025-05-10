import os
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    # Flask app settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    # File paths
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'video_super_resolution', 'models', 'light_deg.pt')
    MODEL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'video_super_resolution', 'models', 'config.json')
    
    # Model settings
    MODEL_SETTINGS = {
        'scale': 4,
        'denoiseLevel': 0,
        'preserveDetails': True,
        'quality': 'balanced',
        'guideScale': 7.5
    }
    
    @staticmethod
    def validate_model_path(model_path: str) -> bool:
        """Validate that the model file exists and is valid"""
        try:
            if not os.path.exists(model_path):
                return False
            
            # Try loading the model
            model = torch.load(model_path, map_location='cpu')
            if not isinstance(model, (dict, torch.nn.Module)):
                return False
                
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_model_config(config_path: str) -> bool:
        """Validate the model configuration file"""
        try:
            if not os.path.exists(config_path):
                return False
                
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Validate required config fields
            required_fields = ['model_type', 'input_channels', 'output_channels']
            return all(field in config for field in required_fields)
        except Exception:
            return False
    
    @staticmethod
    def load_model_config() -> Optional[Dict[str, Any]]:
        """Load and validate model configuration"""
        try:
            if not Config.validate_model_config(Config.MODEL_CONFIG_PATH):
                return None
                
            with open(Config.MODEL_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    @staticmethod
    def init_app(app):
        """Initialize the application with proper validation"""
        # Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        
        # Ensure model directory exists
        model_dir = os.path.dirname(app.config['MODEL_PATH'])
        os.makedirs(model_dir, exist_ok=True)
        
        # Validate model and config
        if not Config.validate_model_path(app.config['MODEL_PATH']):
            raise RuntimeError(f"Invalid or missing model file: {app.config['MODEL_PATH']}")
            
        if not Config.validate_model_config(app.config['MODEL_CONFIG_PATH']):
            raise RuntimeError(f"Invalid or missing model config: {app.config['MODEL_CONFIG_PATH']}")
        
        # Load model config
        model_config = Config.load_model_config()
        if model_config:
            app.config['MODEL_CONFIG'] = model_config
        
        # Log directory paths
        print(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
        print(f"Processed directory: {app.config['PROCESSED_FOLDER']}")
        print(f"Model path: {app.config['MODEL_PATH']}")
        print(f"Model config path: {app.config['MODEL_CONFIG_PATH']}")
        
        # Validate CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.") 