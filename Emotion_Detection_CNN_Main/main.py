import streamlit as st
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import random
import pyttsx3
import csv

# Load the emotion detection model and other necessary data
face_classifier = cv2.CascadeClassifier("C:\\Users\\steve\\Music\\Emotion_Detection_CNN_Main\\Emotion_Detection_CNN_Main\\haarcascade_frontalface_default.xml")
classifier = load_model("C:\\Users\\steve\\Music\\Emotion_Detection_CNN_Main\\Emotion_Detection_CNN_Main\\model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

# Load and parse the quotes CSV
quotes = {}
with open("C:\\Users\\steve\\Music\\Emotion_Detection_CNN_Main\\Emotion_Detection_CNN_Main\\data.csv", 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Skip the header row    
    for row in csv_reader:
        for i, emotion in enumerate(header[1:]):
            quotes.setdefault(emotion, []).append(row[i + 1])
st.title("Emotion-Based Motivational Quotes")

st.write("Welcome to the Emotion-Based Motivational Quotes App!")
st.write("Get ready to improve your mood!")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the TTS engine outside of the loop
engine = pyttsx3.init()

# Add a start button
start_button = st.button("take a live sample")
start_button = st.button("START BUTTON")
# Initialize the emotion and quote variables
current_emotion = None
current_quote = None
selected_quote = None  # Initialize selected_quote outside the loop

# Run emotion detection when the button is clicked
if start_button:
    st.write("Press the button again for the next emotion detection...")

    ret, frame = cap.read()  # Read frame from the camera

    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                max_index = np.argmax(prediction)

                if 0 <= max_index < len(emotion_labels):
                    emotion = emotion_labels[max_index]

                    selected_quote = random.choice(quotes.get(emotion, []))

                    st.write(f"Emotion: {emotion}")
                    st.write(f"Quote: {selected_quote}")

                    label_position = (x, y)
                    cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with the emotion text using Streamlit
        st.image(frame, channels="BGR", use_column_width=True, caption='Emotion Detector')
        
if selected_quote:
    engine.say(selected_quote)
    engine.runAndWait()