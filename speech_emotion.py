from flask import Flask, render_template
from flask_socketio import SocketIO
import speech_recognition as sr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)
socketio = SocketIO(app)

# Load the fine-tuned multilingual transformer model for emotion detection
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Initialize emotion classification pipeline using the loaded model and tokenizer
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Define a label mapping based on the emotion classes
label_mapping = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

# Function to recognize speech from the microphone
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... (Speak for 5 seconds max)")
        
        try:
            # Listen for the first phrase and stop automatically after 5 seconds
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

            # Recognize speech using Google Speech Recognition
            print("Recognizing speech...")
            text = recognizer.recognize_google(audio)
            print(f"Recognized Text: {text}")
            return text
        
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase")
            return None
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

# Function to analyze emotions from recognized speech
def detect_emotion_from_speech(text):
    if text:
        print("Analyzing emotion...")
        emotions = emotion_classifier(text)
        detected_emotions = {}

        for i, emotion in enumerate(emotions[0]):
            emotion_name = label_mapping[i]
            detected_emotions[emotion_name] = emotion['score']

        return detected_emotions

# Handle incoming speech input and start recording on user request
@socketio.on('start_speech_recognition')
def handle_speech_recognition():
    print("Starting speech recognition...")
    recognized_text = recognize_speech_from_mic()
    
    if recognized_text:
        print(f"Recognized text: {recognized_text}")
        emotions = detect_emotion_from_speech(recognized_text)
        print(f"Detected emotions: {emotions}")
        socketio.emit('emotion_result', emotions)
    else:
        print("No speech recognized.")
        socketio.emit('emotion_result', {'error': 'No speech recognized'})

if __name__ == '__main__':
    socketio.run(app, debug=True)