import cv2
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import base64
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from deepface import DeepFace
import torch

app = Flask(__name__)
socketio = SocketIO(app)

# face recognition and emotion detection
def detect_emotion(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if analysis:
            dominant_emotion = analysis[0]['dominant_emotion']
            return dominant_emotion
        return "No Emotion Detected"
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return "Error"

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        emotion = detect_emotion(frame)
        
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Speech Recognition and Emotion Detection
# Load the multilingual transformer model (XLM-RoBERTa) for emotion detection
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, ignore_mismatched_sizes=True)

emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

label_mapping = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

recorded_text = None

def recognize_speech_from_mic(language="en-US"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"Listening... (Language: {language})")
        
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("Recognizing speech...")
            text = recognizer.recognize_google(audio, language=language)
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

def detect_emotion_from_speech(text):
    if text:
        print("Analyzing emotion...")
        emotions = emotion_classifier(text)
        detected_emotions = {}

        for i, emotion in enumerate(emotions[0]):
            emotion_name = label_mapping[i]
            detected_emotions[emotion_name] = emotion['score']

        return detected_emotions

@socketio.on('start_speech_recognition')
def handle_speech_recognition(data):
    global recorded_text
    selected_language = data.get('language', 'en-US')
    print(f"Starting speech recognition with language: {selected_language}")
    recorded_text = recognize_speech_from_mic(selected_language)

@socketio.on('stop_speech_recognition')
def handle_stop_speech_recognition():
    global recorded_text
    print("Stopping speech recognition...")

    if recorded_text:
        print(f"Recognized text: {recorded_text}")
        emotions = detect_emotion_from_speech(recorded_text)
        print(f"Detected emotions: {emotions}")
        socketio.emit('emotion_result', {'text': recorded_text, 'emotions': emotions})
    else:
        print("No speech recognized.")
        socketio.emit('emotion_result', {'error': 'No speech recognized'})

@app.route('/')
def index():
    return render_template('emotion_detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)