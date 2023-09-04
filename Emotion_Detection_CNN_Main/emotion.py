import cv2
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
import random
import pyttsx3
import csv

# Load the emotion detection model and other necessary data
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

# Load and parse the quotes CSV
quotes = {}
with open('data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Skip the header row
    for row in csv_reader:
        for i, emotion in enumerate(header[1:]):
            quotes.setdefault(emotion, []).append(row[i + 1])

# Initialize the TTS engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            max_index = np.argmax(prediction)

            if 0 <= max_index < len(emotion_labels):
                emotion = emotion_labels[max_index]

                # Select a random quote based on the detected emotion
                selected_quote = random.choice(quotes.get(emotion, []))

                label_position = (x, y)
                cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Use the TTS engine to speak the selected quote
                engine.say(selected_quote)
                engine.runAndWait()

        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()