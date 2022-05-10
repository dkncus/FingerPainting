from flask import Flask, render_template, Response
from main import Interpreter
import cv2 as cv

app = Flask(__name__)
interpreter = Interpreter()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Generate each frame
    return Response(gen_frames(interpreter), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(interpreter):
    # Create webcam access object
    camera = cv.VideoCapture(0)

    # Keep reading frames forever
    while True:
        # Read a frame from the camera
        success, frame = camera.read()

        # If there was a frame read from the camera
        if success:

            # Interpret the given frame and detect hand key points
            frame = interpreter.interpret_frame(frame)

            # Encode the image to transmit to the webpage
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Concatenate with header and footer data, then return the frame
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        else:
            break

if __name__ == '__main__':
    app.run()
