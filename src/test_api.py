import requests
import os

def test_api():
    # API endpoint
    url = "http://localhost:8000"
    
    # Test home endpoint
    print("Testing home endpoint...")
    response = requests.get(f"{url}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    response = requests.get(f"{url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test prediction endpoint
    print("\nTesting prediction endpoint...")
    # Replace with path to your test image
    test_image_path = "path/to/your/test/image.jpg"
    
    if os.path.exists(test_image_path):
        with open(test_image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{url}/predict", files=files)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
    else:
        print(f"Test image not found at {test_image_path}")

if __name__ == "__main__":
    test_api() 