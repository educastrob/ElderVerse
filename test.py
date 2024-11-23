#!/usr/bin/env python3

from groq import Groq
from dotenv import load_dotenv
import requests
import os
from flask import Flask, request, jsonify

# Load environment variables from .env file
load_dotenv()

ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
PHONE_NUMBER_ID = os.getenv('PHONE_NUMBER_ID')

WHATSAPP_API_URL = f'https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages'

app = Flask(__name__)

def send_message(to, text):
    print(f"Preparing to send message to: {to}", flush=True)
    headers = {'Authorization': f'Bearer {ACCESS_TOKEN}', 'Content-Type': 'application/json'}
    payload = {'messaging_product': 'whatsapp', 'to': to, 'text': {'body': text}}
    response = requests.post(WHATSAPP_API_URL, json=payload, headers=headers)
    print(f"Message sent from {PHONE_NUMBER_ID} to {to}. Response: {response.json()}", flush=True)
    return response.json()

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    if data and 'messages' in data['entry'][0]['changes'][0]['value']:
        messages = data['entry'][0]['changes'][0]['value']['messages']
        for message in messages:
            phone_number = message['from']
            text = message['text']['body']
            print(f"Received message from {phone_number}: {text}", flush=True)
            # Process the message as needed
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    recipient_number = "+5511993471802"
    message_text = "Hello, this is a test message from ElderVerse!"
    response = send_message(recipient_number, message_text)
    print(f"Response from WhatsApp API: {response}", flush=True)
    ssl_context = ('.certificates/server.crt', '.certificates/server.key')
    app.run(port=5000, debug=True, ssl_context=ssl_context)
