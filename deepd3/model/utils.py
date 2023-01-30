from tensorflow.keras.layers import Input
from tfkerassurgeon import delete_layer, insert_layer

def changeFirstLayer(model):
    new_input = Input(shape=(None, None, 1), name='arbitrary_input')

    model = delete_layer(model.layers[0])
    # inserts before layer 0
    model = insert_layer(model.layers[0], new_input)

    return model
