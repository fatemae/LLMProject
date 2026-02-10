import requests

response = requests.post(
    "http://127.0.0.1:8000/joke/invoke",
    json={'input': {'topic': 'cats'}}
)
json = response.json()
print(json["output"])