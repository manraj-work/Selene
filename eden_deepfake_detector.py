import requests
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get the API key
api = os.getenv("EDEN_AI_KEY")

def detect_local_deepfake(file_path, api):
    headers = {
        "Authorization": "Bearer " + api
    }
    
    url = "https://api.edenai.run/v2/image/ai_detection"
    data = {
        "providers": "winstonai",
    }
    files = {'file': open(file_path, 'rb')}
    
    response = requests.post(url, data=data, files=files, headers=headers)
    return response.json()

# Actually run the function and print results
result = detect_local_deepfake("KendrickMinaj.jpg", api)
print(result)