import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize Pygame mixer for alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haarcascade classifiers for face and eyes detection
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Load trained model
model = load_model('models/cnncat2.h5')

# Start video capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]  # Get frame dimensions

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw a black rectangle for displaying status
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Process right eye detection
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24)) / 255.0  # Normalize pixel values
        r_eye = r_eye.reshape(1, 24, 24, 1)

        rpred = np.argmax(model.predict(r_eye), axis=-1)  # Updated prediction
        break

    # Process left eye detection
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24)) / 255.0  # Normalize pixel values
        l_eye = l_eye.reshape(1, 24, 24, 1)

        lpred = np.argmax(model.predict(l_eye), axis=-1)  # Updated prediction
        break

    # Determine drowsiness status based on eye predictions
    if rpred[0] == 0 and lpred[0] == 0:  # Both eyes closed
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:  # At least one eye open
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Prevent negative score
    if score < 0:
        score = 0

    # Display score
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Trigger alarm if score is too high (continuous eye closure)
    if score > 15:
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        try:
            sound.play()  # Play alarm sound
        except:
            pass

        # Increase or decrease rectangle thickness for alarm effect
        thicc = min(16, thicc + 2) if thicc < 16 else max(2, thicc - 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    # Show the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
