import requests

res = requests.post('https://nlp-assignment-3.onrender.com/test')
print('response from server:', res.text)

dictToSend = {"text": "This majestic predator hunts for prey on the plains of Africa."}
res = requests.post('http://127.0.0.1:5000', json=dictToSend)
print('response from server:', res.text)
dictFromServer = res.json()

dictToSend = {"text": "This majestic predator hunts for prey on the plains of Africa."}
res = requests.post('https://nlp-assignment-3.onrender.com/', json=dictToSend)
print('response from server:', res.text)
dictFromServer = res.json()
