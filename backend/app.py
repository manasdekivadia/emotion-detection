from flask import Flask, render_template, Response
import cv2
from keras.models import model_from_json
import numpy as np
import os
app = Flask(__name__)
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/static'))
app = Flask(__name__, template_folder=template_dir,static_folder=static_dir)
# Load model
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetector.h5")

labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def gen_frames():
    webcam = cv2.VideoCapture(0)
    while True:
        success, frame = webcam.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                except:
                    continue
                img = extract_features(roi_gray)
                prediction = model.predict(img, verbose=0)
                label = labels[prediction.argmax()]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render port or default 5000
    app.run(host='0.0.0.0', port=port, debug=False)