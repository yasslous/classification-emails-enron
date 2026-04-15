import requests

# L'adresse de ton API locale
url = 'http://127.0.0.1:5000/predict'



# Test 
email = {"email": "Hi team, just a reminder that our meeting is scheduled for tomorrow at 10 AM. Thanks!"}

print("--- TEST SPAM ---")
reponse_spam = requests.post(url, json=email)
print(reponse_spam.json())

