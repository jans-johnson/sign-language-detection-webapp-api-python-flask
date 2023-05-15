from flask import Flask, request,jsonify
from flask_cors import CORS
from train import training
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)
count=0
var=training()

@app.route('/upload', methods=['POST'])
def receive_image():
    global count
    # Get the image data from the request payload
    data = request.get_json()
    image_data = data['image']

    # Decode the base64-encoded image data
    image_bytes = base64.b64decode(image_data)

    # Save the image to a file
    with open('frames/'+str(count)+'.jpg', 'wb') as f:
        f.write(image_bytes)
    count=count+1

    # Return a response
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the numpy array into an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if count==30:
        count=0
        x=var.train_model()
        data = {
            'image': base64.b64encode(var.detect(img=img)).decode('utf-8'),
            'message': 'detected : {}'.format(x)
        }
    else:
        data = {
            'image': base64.b64encode(var.detect(img=img)).decode('utf-8'),
            'message': '{} recieved'.format(count)
        }
    
    return jsonify(data)

@app.route('/detect', methods=['POST'])
def detect_image():
    # Get the image data from the request payload
    data = request.get_json()
    image_data = data['image']
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the numpy array into an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Return a response
    return base64.b64encode(var.detect(img=img))

if __name__ == '__main__':
    app.run()