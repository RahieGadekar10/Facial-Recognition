import tensorflow as tf
from tensorflow.keras.layers import Dense , Conv2D, MaxPooling2D , Input, Flatten, Layer
from tensorflow.keras.models import Model
import config
import os
import numpy as np
import pandas as pd
import cv2
from layers import L1Dist

class FaceRecognition :
    def __init__(self, negative_path , positive_path , anchor_path):
        self.NEG_PATH = negative_path
        self.POS_PATH = positive_path
        self.ANC_PATH = anchor_path

    def preprocess(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    def verify(self,model, detection_threshold, verification_threshold):
        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_images', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # Make Predictions
            result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # Detection Threshold: Metric above which a prediciton is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        print(detection)
        # Verification Threshold: Proportion of positive predictions / total positive samples
        #     verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verification = np.median(results)
        print(verification)
        verified = verification > verification_threshold
        return results, verified

def main():
    face = FaceRecognition(config.neg_path , config.pos_path , config.anc_path)
    binary_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-4)
    model = tf.keras.models.load_model("model1.h5", custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = frame[120:100 + 270, 230:200 + 280, :]

        cv2.imshow('Verification', frame)

        # Verification trigger
        if cv2.waitKey(10) & 0xFF == ord('v'):
            cv2.imwrite(os.path.join('application_data', 'input_images', 'input_image.jpg'), frame)
            # Run verification
            results, verified = face.verify(model, 0.8, 0.9)
            print(verified)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__' :
    main()
