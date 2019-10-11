import keras

def amsoftmax_loss(y_true, y_pred, scale = 30, margin = 0.35):
    y_pred = y_pred - y_true*margin
    y_pred = y_pred*scale
    return keras.backend.categorical_crossentropy(y_true, y_pred, from_logits = True)

custom_objects = {'amsoftmax_loss': amsoftmax_loss}
