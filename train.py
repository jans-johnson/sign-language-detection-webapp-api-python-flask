import numpy as np
import mediapipe as mp
import cv2
from tensorflow import keras

class training:

    model = keras.models.load_model('action.h5')
    mp_holistic = mp.solutions.holistic # Holistic model

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
        actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','S','T','U','V','W','X','Y','Z'])
    
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            sequence = []
            for i in range(30):
                frame = cv2.imread('frames/'+str(i)+'.jpg')
                image, results = self.mediapipe_detection(frame, holistic)
                keypoints = self.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
            res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
    
        return actions[np.argmax(res)]
    
    def draw_styled_landmarks(self,image, results):
        mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
    def detect(self,img):
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = self.mediapipe_detection(img, holistic)
            self.draw_styled_landmarks(image, results)
        retval, buffer = cv2.imencode('.jpg', image)
        return buffer

