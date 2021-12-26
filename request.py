import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'V1':-1.23456,'V2':0.12324})

print(r.json())
