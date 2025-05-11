import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize Pygame mixer for alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haarcascade classifiers
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Load trained model
model = load_model('models/cnncat2.h5')

# Start video capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24)) / 255.0
        r_eye = r_eye.reshape(1, 24, 24, 1)

        rpred = np.argmax(model.predict(r_eye), axis=-1)
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24)) / 255.0
        l_eye = l_eye.reshape(1, 24, 24, 1)

        lpred = np.argmax(model.predict(l_eye), axis=-1)
        break

    # Determine eye status
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        status = "Closed"
        color = (0, 0, 255)  # Red
    else:
        score -= 1
        status = "Open"
        color = (0, 255, 0)  # Green

    if score < 0:
        score = 0

    # Display Eye Status
    cv2.putText(frame, f'Eyes: {status}', (10, 30), font, 1, color, 2, cv2.LINE_AA)

    # Draw Score Bar (max score 10)
    bar_width = int((score / 10) * 200)
    cv2.rectangle(frame, (10, height - 30), (210, height - 10), (255, 255, 255), 2)
    cv2.rectangle(frame, (10, height - 30), (10 + bar_width, height - 10), color, -1)
    cv2.putText(frame, f'Score: {score}', (220, height - 15), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Trigger alarm when score > 10
    if score > 10:
        try:
            sound.play()
        except:
            pass

        # Alarm effect rectangle
        thicc = min(16, thicc + 2) if thicc < 16 else max(2, thicc - 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        # Display ALERT text
        cv2.putText(frame, "ALERT!", (width // 2 - 100, height // 2), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
