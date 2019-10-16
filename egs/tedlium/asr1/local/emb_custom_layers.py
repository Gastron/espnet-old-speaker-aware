import keras
import tensorflow as tf
from tensorflow.python.keras import backend as K

def amsoftmax_loss(y_true, y_pred, scale = 30, margin = 0.35):
    y_pred = y_pred - y_true*margin
    y_pred = y_pred*scale
    return keras.backend.categorical_crossentropy(y_true, y_pred, from_logits = True)

class VLAD(keras.engine.Layer):

    def __init__(self, k_centers=8, kernel_initializer='glorot_uniform', **kwargs):
        self.k_centers = k_centers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        super(VLAD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='kernel', 
                                      shape=(input_shape[2], self.k_centers),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.b = self.add_weight(name='kernel', 
                                      shape=(self.k_centers, ),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.c = self.add_weight(name='kernel', 
                                      shape=(input_shape[2], self.k_centers),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        super(VLAD, self).build(input_shape)  

    def call(self, x):
        
        Wx_b = K.dot(x, self.w)+self.b
        a = tf.nn.softmax(Wx_b)
        
        rows = []

        for k in range(self.k_centers):
            error = x-self.c[:, k]
            
            row = K.batch_dot(a[:, :, k],error)
            row = tf.nn.l2_normalize(row,dim=1)
            rows.append(row)
            
        output = tf.stack(rows)
        output = tf.transpose(output, perm = [1, 0, 2])
        output = tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[1]*tf.shape(output)[2]])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k_centers*input_shape[2])
    
    def get_config(self):
        config = super(VLAD, self).get_config()
        config['k_centers'] = self.k_centers
        config['kernel_initializer'] = initializers.serialize(self.kernel_initializer)
        return config

custom_objects = {'VLAD': VLAD, 'amsoftmax_loss': amsoftmax_loss}
