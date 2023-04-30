from flask import Flask, request
from flask_cors import CORS
from train import training

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
    import base64
    image_bytes = base64.b64decode(image_data)

    # Save the image to a file
    with open('frames/'+str(count)+'.jpg', 'wb') as f:
        f.write(image_bytes)
    count=count+1
    if count==30:
        count=0
        x=var.train_model()
        return 'detected : {}'.format(x)
    # Return a response
    return '{} recieved'.format(count)

if __name__ == '__main__':
    app.run()