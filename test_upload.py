import requests
import json
import os
import time

def test_video_upload(video_path, settings=None):
    if settings is None:
        settings = {
            'scale': 4,
            'denoiseLevel': 0,
            'preserveDetails': True,
            'quality': 'balanced',
            'guideScale': 7.5  # Make sure this matches the backend expectation
        }

    # API endpoint
    url = 'http://localhost:5000/api/upload'

    # Prepare the files and data
    files = {
        'video': ('test.mp4', open(video_path, 'rb'), 'video/mp4')
    }
    
    data = {
        'settings': json.dumps(settings)
    }

    try:
        # Send POST request
        print("Uploading video...")
        print(f"Using settings: {settings}")  # Print settings for debugging
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get('taskId')
            print(f"Upload successful! Task ID: {task_id}")
            
            # Monitor task status with detailed progress
            while True:
                progress_response = requests.get(f'http://localhost:5000/api/progress/{task_id}')
                if progress_response.status_code == 200:
                    progress_data = progress_response.json()
                    print(f"Status: {progress_data.get('status')}, Progress: {progress_data.get('percentage', 0):.2f}%, Frame: {progress_data.get('current_frame', 0)}/{progress_data.get('total_frames', '?')}, ETA: {progress_data.get('estimated_time_remaining', '?')}s")
                    if progress_data.get('status') in ['complete', 'error']:
                        if progress_data.get('status') == 'complete':
                            print("Processing completed successfully!")
                            # Optionally, get download URL from /api/status
                            status_response = requests.get(f'http://localhost:5000/api/status/{task_id}')
                            if status_response.status_code == 200:
                                status_data = status_response.json()
                                print(f"Download URL: {status_data.get('downloadUrl')}")
                        else:
                            print(f"Error: {progress_data.get('error')}")
                        break
                    time.sleep(2)
                else:
                    print(f"Error checking progress: {progress_response.text}")
                    break
        else:
            print(f"Upload failed: {response.text}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Close the file
        files['video'][1].close()

if __name__ == "__main__":
    # Use the correct video path
    video_path = "input/video/023_klingai_reedit.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Test video file '{video_path}' not found!")
    else:
        test_video_upload(video_path) 