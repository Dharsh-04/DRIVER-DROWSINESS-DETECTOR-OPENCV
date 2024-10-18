import cv2
import numpy as np
from keras.models import load_model
import winsound  # For playing sound on Windows
import threading
import time

# Load the pre-trained model
model_path = 'models/cnnCat2.keras'  # Adjust path if necessary
model = load_model(model_path)
print("Model loaded successfully.")

# Load Haar Cascades
left_eye_cascade = cv2.CascadeClassifier(
    'G:\\DHARSHNI_WORKS\\Drowsy_driver_detector\\haar cascade files\\haarcascade_lefteye_2splits.xml'
)
right_eye_cascade = cv2.CascadeClassifier(
    'G:\\DHARSHNI_WORKS\\Drowsy_driver_detector\\haar cascade files\\haarcascade_righteye_2splits.xml'
)

# Alarm state and timing variables
alarm_on = False
eyes_closed_start = None
eyes_not_detected_count = 0  # Counter for eyes not detected

def play_alarm():
    global alarm_on
    if not alarm_on:
        alarm_on = True
        print("Playing alarm...")
        winsound.PlaySound('G:\\DHARSHNI_WORKS\\Drowsy_driver_detector\\sound\\alarm.wav', winsound.SND_LOOP + winsound.SND_ASYNC)

def stop_alarm():
    global alarm_on
    if alarm_on:
        print("Stopping alarm...")
        winsound.PlaySound(None, winsound.SND_PURGE)
        alarm_on = False

def preprocess_image(roi):
    roi = cv2.resize(roi, (24, 24))
    roi = roi.astype('float32') / 255.0  # Normalize
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension
    return roi

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_eye = left_eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    right_eye = right_eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    eyes_open = len(left_eye) > 0 and len(right_eye) > 0

    drowsiness_prob = 0.0  # Initialize drowsiness probability

    if eyes_open:
        print("Eyes detected.")  # Debug message
        x, y, w, h = left_eye[0]
        left_eye_frame = preprocess_image(frame[y:y+h, x:x+w])

        x, y, w, h = right_eye[0]
        right_eye_frame = preprocess_image(frame[y:y+h, x:x+w])

        left_pred = model.predict(left_eye_frame)[0][0]
        right_pred = model.predict(right_eye_frame)[0][0]
        drowsiness_prob = (left_pred + right_pred) / 2  # Average predictions

        if drowsiness_prob > 0.5:  # Drowsy case
            if eyes_closed_start is None:
                eyes_closed_start = time.time()
        else:  # Reset alarm and timer
            eyes_closed_start = None
            stop_alarm()

        eyes_not_detected_count = 0  # Reset counter if eyes are detected

    else:
        print("Eyes not detected.")  # Debug message
        eyes_not_detected_count += 1  # Increment the counter
        if eyes_not_detected_count > 4:  # More than 4 frames without detecting eyes
            threading.Thread(target=play_alarm).start()

    # Check if eyes are closed for more than 2 seconds
    if eyes_closed_start and (time.time() - eyes_closed_start >= 2):
        threading.Thread(target=play_alarm).start()
        label = "Drowsy"
        color = (0, 0, 255)  # Red
    else:
        label = "Alert"
        color = (0, 255, 0)  # Green

    # Display status on the frame
    cv2.putText(frame, f"{label}: {drowsiness_prob:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
