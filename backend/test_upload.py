import requests
import json
import time

# Video file path
video_path = 'input/video/test.mp4'

# Settings
settings = {
    'scale': 4,
    'quality': 'balanced',
    'denoiseLevel': 0,
    'preserveDetails': True
}

# Create the multipart form data
files = {
    'video': ('test.mp4', open(video_path, 'rb'), 'video/mp4')
}
data = {
    'settings': json.dumps(settings)
}

# Make the request
response = requests.post('http://localhost:5000/api/upload', files=files, data=data)

# Print the response
print(f'Status code: {response.status_code}')
print(f'Response: {response.json()}')

# Auto-polling for status
if response.status_code == 200 and 'taskId' in response.json():
    task_id = response.json()['taskId']
    status_url = f'http://localhost:5000/api/status/{task_id}'
    print(f'Polling status at: {status_url}')
    while True:
        status_resp = requests.get(status_url)
        status_json = status_resp.json()
        print(f"Status: {status_json.get('progress', 'unknown')}%")
        if status_json.get('downloadUrl'):
            print(f"Processing complete! Download URL: {status_json['downloadUrl']}")
            break
        if status_json.get('error'):
            print(f"Error: {status_json['error']}")
            break
        time.sleep(5) 