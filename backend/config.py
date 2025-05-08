import os

class Config:
    FLASK_APP = 'app.py'
    FLASK_ENV = 'development'
    FLASK_DEBUG = True
    MAX_CONTENT_LENGTH = 524288000  # 500MB in bytes
    UPLOAD_FOLDER = 'input/video'
    PROCESSED_FOLDER = 'output/video'
    MODEL_PATH = 'pretrained_weight/light_deg.pt'
    
    @staticmethod
    def init_app(app):
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.PROCESSED_FOLDER, exist_ok=True) 