import requests
import time

# res = requests.post('https://nlp-assignment-3.onrender.com/test')
# print('response from server:', res.text)

# start_time = time.time()
# dictToSend = {"text": "This is a red, round fruit."}
# res = requests.post('https://nlp-assignment-3.onrender.com/predict', json=dictToSend)
# print('response from server:', res.text)
# end_time = time.time()

start_time = time.time()
dictToSend = {"text": "This is a red, round fruit."}
res = requests.post('http://127.0.0.1:5000/predict', json=dictToSend)
print('response from server:', res.text)
end_time = time.time()
