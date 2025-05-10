import requests
import json

# Video file path
video_path = 'input/video/023_klingai_reedit.mp4'

# Settings
settings = {
    'scale': 4,
    'quality': 'balanced',
    'denoiseLevel': 0,
    'preserveDetails': True
}

# Create the multipart form data
files = {
    'video': ('023_klingai_reedit.mp4', open(video_path, 'rb'), 'video/mp4')
}
data = {
    'settings': json.dumps(settings)
}

# Make the request
response = requests.post('http://localhost:5000/api/upload', files=files, data=data)

# Print the response
print(f'Status code: {response.status_code}')
print(f'Response: {response.json()}') 