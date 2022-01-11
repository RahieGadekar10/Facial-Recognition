import tensorflow as tf
from tensorflow.keras.layers import Dense , Conv2D, MaxPooling2D , Input, Flatten, Layer
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from tensorflow.keras.models import Model
import config
import os
import numpy as np
import pandas as pd
import cv2
tf.config.run_functions_eagerly(True)

from layers import L1Dist

class train :
    def __init__(self, negative_path , positive_path , anchor_path):
        self.NEG_PATH = negative_path
        self.POS_PATH = positive_path
        self.ANC_PATH = anchor_path

    def create_data(self):
        anchor = tf.data.Dataset.list_files(self.ANC_PATH + '\*.jpg').take(1000) #Change the value according the number of images("Length of all the three variables should be same")
        positive = tf.data.Dataset.list_files(self.POS_PATH + '\*.jpg').take(1000)
        negative = tf.data.Dataset.list_files(self.NEG_PATH + '\*.jpg').take(1000)

        positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(list(anchor))))))
        negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(list(anchor))))))
        data = positives.concatenate(negatives)
        return data


    def preprocess(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img


    def preprocess_twin(self, input_img, valid_img, label):
        return (self.preprocess(input_img), self.preprocess(valid_img), label)


    def training_partition(self, data):
        train_data = data.take(round(len(list(data)) * 0.7))
        train_data = train_data.batch(16)
        train_data = train_data.prefetch(8)
        return train_data


    def testing_partition(self, data):
        test_data = data.skip(round(len(list(data)) * 0.7))
        test_data = test_data.take(round(len(list(data)) * 0.3))
        test_data = test_data.batch(16)
        test_data = test_data.prefetch(8)
        return test_data


    def embedding_layer(self):
        inp = Input((100, 100, 3), name="input_image")

        # First Block
        con1 = Conv2D(64, (10, 10), activation='relu')(inp)
        max1 = MaxPooling2D(64, (2, 2), padding='same')(con1)

        # Second Block
        con2 = Conv2D(128, (7, 7), activation='relu')(max1)
        max2 = MaxPooling2D(64, (2, 2), padding='same')(con2)

        # Thrid Block
        con3 = Conv2D(128, (4, 4), activation='relu')(max2)
        max3 = MaxPooling2D(64, (2, 2), padding='same')(con3)

        # Final Block
        con4 = Conv2D(256, (4, 4), activation='relu')(max3)
        f1 = Flatten()(con4)
        op = Dense(4096, activation='sigmoid')(f1)

        return Model(inputs=[inp], outputs=[op], name='embedding_layer')


    def make_siamese(self, embedding):
        input_embedding = Input((100, 100, 3), name='input_image')
        valid_embedding = Input((100, 100, 3), name='valid_image')

        siamese_network = L1Dist()

        distances = siamese_network(embedding(input_embedding), embedding(valid_embedding))

        classifier = Dense(1, activation='sigmoid')(distances)

        return Model(inputs=[input_embedding, valid_embedding], outputs=classifier, name='siamese_network')

    @tf.function
    def train_step(self,model , batch ):
        binary_loss = tf.losses.BinaryCrossentropy()
        opt = tf.keras.optimizers.Adam(1e-4)
        with tf.GradientTape() as tape:
            print("Running")
            # Get anchor and positive/negative image
            x = batch[:2]
            # Get label
            y = batch[2]
            # Forward propagation
            yhat = model(x, training=True)
            # Calculate loss
            loss = binary_loss(y, yhat)
            print("Running")
        print(loss)

        # Calculate gradient
        grad = tape.gradient(loss, model.trainable_variables)
        print("Running")

        # Calculate updated weights and apply to the siamese model
        opt.apply_gradients(zip(grad, model.trainable_variables))
        return loss


    def train(self,data, model, epochs):
        # loop through epochs
        for epoch in range(1, epochs + 1):
            print(f'\n Epoch {epoch} / {epochs}')
            progbar = tf.keras.utils.Progbar(len(data))

            # Creating a metric object
            r = Recall()
            p = Precision()
            a = Accuracy()

            # Loop through each batch
            for idx, batch in enumerate(data):
                # Run train step here
                loss = self.train_step(model,batch)
                yhat = model.predict(batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                a.update_state(batch[2], yhat)
                progbar.update(idx + 1)
            print(loss.numpy(), r.result().numpy(), p.result().numpy(), a.result().numpy())

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    face = train(config.neg_path , config.pos_path , config.anc_path)

    # Build a dataloader pipeline
    data = face.create_data()
    data = data.map(face.preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=2000)
    train_data = face.training_partition(data)
    test_data = face.testing_partition(data)
    embedding = face.embedding_layer()
    siamese = face.make_siamese(embedding)
    # face.train_step(siamese)
    face.train(train_data ,siamese,10)

    siamese.save("model.h5")

if __name__ == '__main__' :
    main()