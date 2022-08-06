import tensorflow as tf
from tensorflow import keras
import triplet_utils
from tensorflow.keras import layers
import json
import numpy as np

def fc_blocks(inputs, sizes, dropout = 0.0):
    x = inputs
    for size in sizes:
        x = fc_block(x, size, dropout)
    return x

def fc_block(inputs, size, dropout):
    x = layers.Dense(size, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if (dropout > 0.0):
        x = layers.Dropout(dropout)(x)

    return x

class SimpleClassifier():
    def __init__(self, filename, target_size):
        self.model = keras.models.load_model(filename)
        self.labels = None
    
    def set_labels(self, json_filename):
        with open(json_filename, 'r') as fp:
            self.labels = json.load(fp)
    
    def predict_likelihoods(self, image):
        pred_raw = self.model.predict(tf.expand_dims(image, 0))
        return pred_raw

    def predict(self, image):
        likelihoods = self.predict_likelihoods(image)
        pred_raw = np.argmax(likelihoods)
        return pred_raw if self.labels is None else self.labels[str(pred_raw)]
    
class SiameseClassifier():
    def __init__(self, filename, target_size):
        embedding_size = 128
        base_network = tf.keras.applications.InceptionResNetV2(include_top=False,
                                         weights=None,
                                         input_shape=(128, 128, 3),
                                         pooling='avg')

        # Freeze the base model
        base_network.trainable = False

        dummy_input = keras.Input((128, 128, 3))
        base_network = base_network(dummy_input, training=False)

        x = fc_blocks(base_network, sizes=[512, 256], dropout=0.2)

        embedding_layer = layers.Dense(embedding_size,
                                       activation=None)(x)
        base_network = keras.Model(dummy_input,
                                   embedding_layer,
                                   name='InceptionResNetV2')
        print("Create base network with layers:")
        for layer in base_network.layers:
            print("Name: %s | Trainable: %s" % (layer.name, layer.trainable))
        input_anchor = keras.Input(target_size, name='Anchor')
        input_positive = keras.Input(target_size, name='Positive')
        input_negative = keras.Input(target_size, name='Negative')
        embedding_anchor = base_network(input_anchor)
        embedding_positive = base_network(input_positive)
        embedding_negative = base_network(input_negative)
        margin = 1

        triplet_loss_layer = triplet_utils.TripletLossLayer(alpha=margin, name='triplet_loss_layer')([embedding_anchor, embedding_positive, embedding_negative])
        breed_loss_layer = layers.Dense(37, activation='softmax', name='breed')(embedding_anchor)
        species_loss_layer = layers.Dense(2, activation='softmax', name='species')(embedding_anchor)

        triplet_network = keras.Model(inputs=[input_anchor, input_positive, input_negative],
                                      outputs=[species_loss_layer, breed_loss_layer, triplet_loss_layer],
                                      )
        print("Loading weights...")
        triplet_network.load_weights(filename)
        self.model = triplet_network
    
    def predict(self, image):
        species, breed, loss_pred = self.model.predict([tf.expand_dims(img, 0), tf.expand_dims(img, 0),tf.expand_dims(img, 0)], verbose=False)
        return species, breed
