import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from gtts import gTTS
from playsound import playsound
import os
import time

# Load model and label map
model = tf.keras.models.load_model("sign_model.h5")
with open("label_map.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

last_prediction = ""
last_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x_data = []
            y_data = []

            for lm in handLms.landmark:
                x_data.append(lm.x)
                y_data.append(lm.y)

            input_data = np.array(x_data + y_data).reshape(1, -1)
            prediction = model.predict(input_data)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            # Only speak if it's a new word and 2 seconds passed
            if predicted_label != last_prediction and time.time() - last_time > 2:
                print("You signed:", predicted_label)
                
                import uuid
                tts = gTTS(text=predicted_label, lang='en')
                filename = f"{uuid.uuid4()}.mp3"
                tts.save(filename)
                
                import time
                time.sleep(0.5)
                playsound(filename)
                os.remove(filename)# For Windows
                last_prediction = predicted_label
                last_time = time.time()

            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign to Speech", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()