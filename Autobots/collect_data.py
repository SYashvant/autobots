import cv2
import mediapipe as mp
import csv
import os
import time

# Settings
label = input("Enter the label for this sign (e.g. hello, yes, no): ")
samples = int(input("How many samples to collect? "))

# Create folder if not exists
if not os.path.exists("data"):
    os.makedirs("data")

filename = f"data/{label}.csv"
file = open(filename, mode='w', newline='')
csv_writer = csv.writer(file)
csv_writer.writerow([f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"])

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

print("Starting collection in 5 seconds... Get ready!")
time.sleep(5)
print("Collecting...")

count = 0
while count < samples:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x_data = []
            y_data = []
            for lm in handLms.landmark:
                x_data.append(lm.x)
                y_data.append(lm.y)

            row = x_data + y_data + [label]
            csv_writer.writerow(row)
            count += 1
            print(f"Collected sample {count}/{samples}")

    cv2.imshow("Collecting Data", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()
print("Done collecting data.")