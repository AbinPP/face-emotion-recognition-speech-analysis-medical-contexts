import cv2
from deepface import DeepFace

def detect_emotion(frame):
    try:
        # Analyze the frame for emotion detection
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Extract the dominant emotion
        if analysis:
            dominant_emotion = analysis[0]['dominant_emotion']
            return dominant_emotion
        else:
            return "No Emotion Detected"
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return "Error"

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Detect emotion
    emotion = detect_emotion(frame)

    if emotion:
        # Display the emotion on the frame
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

