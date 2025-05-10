import requests
import time

task_id = '52cacce1-36e6-49e2-a18c-40f5629aebc9'
url = f'http://localhost:5000/api/status/{task_id}'

while True:
    try:
        response = requests.get(url)
        print(f'Status code: {response.status_code}')
        print(f'Response: {response.json()}')
        
        if response.status_code == 200:
            status = response.json().get('status')
            if status == 'completed':
                print('Processing completed!')
                break
            elif status == 'failed':
                print('Processing failed!')
                break
        
        time.sleep(5)  # Wait 5 seconds before checking again
        
    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')
        break 