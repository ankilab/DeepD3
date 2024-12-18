import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPool2D, \
    UpSampling2D, BatchNormalization, Activation, Add, Layer

class InstanceNormalization(Layer):
    """Custom Instance Normalization Layer"""
    def __init__(self, epsilon=1e-5, **kwargs):  # Added **kwargs to handle additional parameters like name
        super(InstanceNormalization, self).__init__(**kwargs)  # Pass kwargs to parent class
        self.epsilon = epsilon

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale = self.add_weight(
            name='scale',
            shape=[depth],
            initializer='ones',
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=[depth],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def convlayer(x, filters, activation, name, residual=None, use_instancenorm=True):
    """Convolutional layer with normalization and residual connection

    Args:
        x (Keras.layer): input layer
        filters (int): filters used in convolutional layer
        activation (str): Activation function
        name (str): Description of layer
        residual (Keras.layer, optional): Residual layer. Defaults to None.
        use_instancenorm (bool): Use of instance normalization. Defaults to True.

    Returns:
        Keras.layer: Full convolutional procedure
    """
    x = Conv2D(filters, 3, padding='same', use_bias=False, name=name)(x)

    if use_instancenorm:
        x = InstanceNormalization(name=name+"_InstanceNorm")(x)

    if type(residual) is not type(None):
        x = Add(name=name+"_residual_connection")([residual, x])  # Fixed tuple syntax

    x = Activation(activation, name=name+"_activation")(x)
    return x

def identity(x, filters, name):
    """Identity layer for residual layers

    Args:
        x (Keras.layer): Keras layer
        filters (int): Used filters
        name (str): Layer description

    Returns:
        Keras.layer: Identity layer
    """
    return Conv2D(filters, 1, padding='same', use_bias=False, name=name)(x)

def decoder(x, filters, layers, to_concat, name, activation):
    """Decoder for neural network.

    Args:
        x (Keras layer): Start of decoder, normally the latent space
        filters (int): The filter multiplier 
        layers (int): Depth layers to be used for upsampling
        to_concat (list): Encoder layers to be concatenated 
        name (str): Description of the decoder
        activation (str): Activation function used in Decoder

    Returns:
        Keras layer: Full decoder across layers
    """
    # Decoder
    for i in range(layers):
        # Upsamples the current activation maps by a factor of 2x2
        x = UpSampling2D()(x)

        # Concatenates respective encoder layer
        x = Concatenate()([x, to_concat.pop()])

        # Applies two convolutional layers
        x = convlayer(x, filters*2**(layers-1-i), activation, f"{name}_dec_layer{layers-i}_conv1")
        x = convlayer(x, filters*2**(layers-1-i), activation, f"{name}_dec_layer{layers-i}_conv2")

    # Final point-wise convolution to achieve prediction maps
    x = Conv2D(1, 1, padding='same', name=name, activation='sigmoid')(x)

    return x
    
def DeepD3_Model(filters=32, input_shape=(128, 128, 1), layers=4, activation="swish"):
    """DeepD3 TensorFlow Keras Model. It defines the architecture,
    together with the single encoder and dual decoders.

    Args:
        filters (int, optional): Base filter multiplier. Defaults to 32.
        input_shape (tuple, optional): Image shape for training. Defaults to (128, 128, 1).
        layers (int, optional): Network depth layers. Defaults to 4.
        activation (str, optional): Activation function used in convolutional layers. Defaults to "swish".

    Returns:
        Model: function TensorFlow/Keras model
    """
    # Save concatenation layers
    to_concat = []
    
    # Create model input
    model_input = Input(input_shape, name="input")
    x = model_input
    
    # Common Encoder
    for i in range(layers):
        residual = identity(x, filters*2**i, f"enc_layer{i}_identity")
        x = convlayer(x, filters*2**i, activation, f"enc_layer{i}_conv1")
        x = convlayer(x, filters*2**i, activation, residual=residual, name=f"enc_layer{i}_conv2")
        to_concat.append(x)
        x = MaxPool2D()(x)
        
    # Latent
    x = convlayer(x, filters*2**(i+1), activation, f"latent_conv")
    
    # Two decoder, for dendrites and spines each
    dendrites = decoder(x, filters, layers, to_concat.copy(), "dendrites", activation)
    spines = decoder(x, filters, layers, to_concat.copy(), "spines", activation)

    return Model(model_input, [dendrites, spines])

if __name__ == '__main__':
    # Test if model creation and training works
    import numpy as np

    # Create Model
    m = DeepD3_Model(8, input_shape=(48,48,1))

    # Create a random dataset of 100 images of tile size 48x48
    X = np.random.randn(48*48*100).reshape(100, 48, 48, 1)

    print(m.summary())

    # Prepare and fit for one epoch
    m.compile('sgd',['mae','mse'])
    m.fit(X, [X, X], epochs=1)