import requests
import json

def test_prediction():
    url = "http://localhost:8000/predict"
    payload = {
        "ticker": "MSFT",
        "days_to_predict": 5
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Success!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure the API is running (python api.py)")

if __name__ == "__main__":
    test_prediction()
