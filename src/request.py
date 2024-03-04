import requests

# Assuming your BentoML server is running on http://localhost:5000
api_url = 'http://localhost:5000/lstm_classifier/classify'

# Example input data
input_data = {"comment": "This movie is great !!! everyone should match this movie at least once!."}
print(input_data)
# Make a POST request to the BentoML API endpoint
response = requests.get(api_url, json=input_data)
print(input_data)
# Check the response
if response.status_code == 200:
    result = response.json()
    print(f"Classification result: {result}")
else:
    print(f"Failed with status code: {response.status_code}, message: {response.text}")
