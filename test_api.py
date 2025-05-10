import requests
import sys

def test_api():
    try:
        response = requests.get('http://localhost:5000/api/test')
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api() 