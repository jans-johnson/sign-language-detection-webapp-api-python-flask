import numpy as np
import mediapipe as mp
import cv2
from tensorflow import keras

class training:

    model = keras.models.load_model('action.h5')

    def mediapipe_detection(self,image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def extract_keypoints(self,results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])

    def train_model(self):
        mp_holistic = mp.solutions.holistic # Holistic model
        actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','S','T','U','V','W','X','Y','Z'])
    
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            sequence = []
            for i in range(30):
                frame = cv2.imread('frames/'+str(i)+'.jpg')
                image, results = self.mediapipe_detection(frame, holistic)
                keypoints = self.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
            res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
    
        return actions[np.argmax(res)]