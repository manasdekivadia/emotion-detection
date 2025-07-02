import cv2
from keras.models import model_from_json
import numpy as np

# Load model architecture
with open("../emotiondetector.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load weights
model.load_weights("../emotiondetector.h5")

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocessing function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Model expects shape (1, 48, 48, 1)
    return feature / 255.0

# Webcam setup
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

while True:
    success, frame = webcam.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        try:
            roi_gray = cv2.resize(roi_gray, (48, 48))
        except:
            continue  # Skip if resize fails

        img = extract_features(roi_gray)
        prediction = model.predict(img, verbose=0)
        label = labels[prediction.argmax()]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show output frame
    cv2.imshow("Emotion Detector", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
