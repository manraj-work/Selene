import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()
api_key = os.getenv("HUGGING_FACE_API_KEY")

def detect_deepfake(image_path):
    # Using a popular, actively maintained model
    url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    with open(image_path, "rb") as f:
        data = f.read()
    
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        # This model works - now let's find a deepfake one
        print("✓ API is working! Now trying deepfake models...")
        
        # Try a different deepfake model
        deepfake_models = [
            "jlbaker361/deepfake-detection",
            "chanelcolgate/deepfake-detection", 
            "kjdhfkjdh/deepfake_detection"
        ]
        
        for model in deepfake_models:
            print(f"\nTrying {model}...")
            url = f"https://api-inference.huggingface.co/models/{model}"
            response = requests.post(url, headers=headers, data=data)
            
            if response.status_code == 200:
                return f"Success with {model}: {response.json()}"
            else:
                print(f"Failed: {response.status_code}")
    
    return "No working deepfake models found"

result = detect_deepfake("KendrickMinaj.jpg")
print(result)