from tensorflow.keras.layers import Layer
import tensorflow as tf

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, valid_embedding):
        return tf.math.abs(input_embedding - valid_embedding)