import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPool2D, \
    UpSampling2D, LayerNormalization, Activation, Add

def convlayer(x, filters, activation, name, residual=None, use_layernorm=True):
    """Convolutional layer with layer normalization and residual connection

    Args:
        x (Keras.layer): input layer
        filters (int): filters used in convolutional layer
        activation (str): Activation function
        name (str): Description of layer
        residual (Keras.layer, optional): Residual layer. Defaults to None.
        use_layernorm (bool): Use of layer normalization. Defaults to True.

    Returns:
        Keras.layer: Full convolutional procedure
    """
    x = Conv2D(filters, 3, padding='same', use_bias=False, name=name)(x)

    if use_layernorm:
        # Layer normalization applied after convolution
        x = LayerNormalization(
            axis=[1, 2, 3],  # Normalize across all dimensions except batch
            epsilon=1e-5,
            center=True,  # Enable bias
            scale=True,   # Enable scale
            name=name+"_LayerNorm"
        )(x)

    if residual is not None:
        x = Add(name=name+"_residual_connection")([residual, x])

    x = Activation(activation, name=name+"_activation")(x)
    return x

def identity(x, filters, name):
    """Identity layer for residual layers"""
    return Conv2D(filters, 1, padding='same', use_bias=False, name=name)(x)

def decoder(x, filters, layers, to_concat, name, activation):
    """Decoder for neural network."""
    for i in range(layers):
        x = UpSampling2D()(x)
        x = Concatenate()([x, to_concat.pop()])
        x = convlayer(x, filters*2**(layers-1-i), activation, f"{name}_dec_layer{layers-i}_conv1")
        x = convlayer(x, filters*2**(layers-1-i), activation, f"{name}_dec_layer{layers-i}_conv2")

    # Final layer without normalization for preserving raw predictions
    x = Conv2D(1, 1, padding='same', name=name, activation='sigmoid')(x)
    return x
    
def DeepD3_Model(filters=32, input_shape=(128, 128, 1), layers=4, activation="swish"):
    """DeepD3 TensorFlow Keras Model with Layer Normalization"""
    to_concat = []
    
    model_input = Input(input_shape, name="input")
    x = model_input
    
    # Encoder with Layer Normalization
    for i in range(layers):
        residual = identity(x, filters*2**i, f"enc_layer{i}_identity")
        x = convlayer(x, filters*2**i, activation, f"enc_layer{i}_conv1")
        x = convlayer(x, filters*2**i, activation, residual=residual, name=f"enc_layer{i}_conv2")
        to_concat.append(x)
        x = MaxPool2D()(x)
        
    # Latent space
    x = convlayer(x, filters*2**(i+1), activation, f"latent_conv")
    
    # Decoders
    dendrites = decoder(x, filters, layers, to_concat.copy(), "dendrites", activation)
    spines = decoder(x, filters, layers, to_concat.copy(), "spines", activation)

    return Model(model_input, [dendrites, spines])

if __name__ == '__main__':
    # Test implementation
    import numpy as np
    
    # Create test model
    m = DeepD3_Model(32, input_shape=(128, 128, 1))
    
    # Print model summary
    print(m.summary())
    
    # Test with random data
    X = np.random.randn(10, 128, 128, 1)  # 10 test images
    m.compile('adam', ['dice_loss', 'mse'])
    predictions = m.predict(X)