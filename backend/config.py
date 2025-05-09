import os

class Config:
    # Flask app settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    # File paths
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'video_super_resolution', 'models', 'star.pth')
    
    @staticmethod
    def init_app(app):
        # Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        
        # Ensure model directory exists
        model_dir = os.path.dirname(app.config['MODEL_PATH'])
        os.makedirs(model_dir, exist_ok=True)
        
        # Log directory paths
        print(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
        print(f"Processed directory: {app.config['PROCESSED_FOLDER']}")
        print(f"Model path: {app.config['MODEL_PATH']}") 