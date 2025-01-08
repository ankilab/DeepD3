import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
tfd = tfp.distributions

class DownBlock(layers.Layer):
    def __init__(self, filters, name="down_block"):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.pool = layers.MaxPooling2D()
        
    def call(self, x):
        skip = self.conv1(x)
        skip = self.conv2(skip)
        x = self.pool(skip)
        return x, skip

class UpBlock(layers.Layer):
    def __init__(self, filters, name="up_block"):
        super().__init__(name=name)
        self.up = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')
        self.conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters, 3, padding='same', activation='relu')
        
    def call(self, x, skip):
        x = self.up(x)
        x = tf.concat([x, skip], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Encoder(layers.Layer):
    def __init__(self, filters, name="encoder"):
        super().__init__(name=name)
        self.down1 = DownBlock(filters)
        self.down2 = DownBlock(filters * 2)
        self.down3 = DownBlock(filters * 4)
        self.down4 = DownBlock(filters * 8)
        
    def call(self, x):
        skips = []
        x, skip1 = self.down1(x)
        skips.append(skip1)
        x, skip2 = self.down2(x)
        skips.append(skip2)
        x, skip3 = self.down3(x)
        skips.append(skip3)
        x, skip4 = self.down4(x)
        skips.append(skip4)
        return x, skips

class Decoder(layers.Layer):
    def __init__(self, filters, name="decoder"):
        super().__init__(name=name)
        self.up1 = UpBlock(filters * 8)
        self.up2 = UpBlock(filters * 4)
        self.up3 = UpBlock(filters * 2)
        self.up4 = UpBlock(filters)
        self.final = layers.Conv2D(1, 1, activation='sigmoid')
        
    def call(self, x, skips):
        x = self.up1(x, skips[3])
        x = self.up2(x, skips[2])
        x = self.up3(x, skips[1])
        x = self.up4(x, skips[0])
        return self.final(x)

class Prior(layers.Layer):
    def __init__(self, latent_dim, name="prior"):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        
    def call(self, batch_size):
        mu = tf.zeros([batch_size, self.latent_dim])
        sigma = tf.ones([batch_size, self.latent_dim])
        return tfd.Normal(mu, sigma)

class Posterior(layers.Layer):
    def __init__(self, latent_dim, name="posterior"):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.pool = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.dense_mu = layers.Dense(latent_dim)
        self.dense_sigma = layers.Dense(latent_dim, activation='softplus')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        mu = self.dense_mu(x)
        sigma = self.dense_sigma(x)
        return tfd.Normal(mu, sigma)

class DualOutputProbabilisticUNet(Model):
    def __init__(self, input_shape, filters=32, latent_dim=6):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Shared encoder
        self.encoder = Encoder(filters)
        
        # Separate decoders for dendrites and spines
        self.decoder_dendrites = Decoder(filters, name="dendrite_decoder")
        self.decoder_spines = Decoder(filters, name="spine_decoder")
        
        # Prior network
        self.prior = Prior(latent_dim)
        
        # Separate posteriors for dendrites and spines
        self.posterior_dendrites = Posterior(latent_dim, name="dendrite_posterior")
        self.posterior_spines = Posterior(latent_dim, name="spine_posterior")
        
    def call(self, inputs, training=False):
        if training:
            images, (masks_dendrites, masks_spines) = inputs
        else:
            images = inputs
            
        # Encode image
        encoded, skips = self.encoder(images)
        
        if training:
            # During training, use posterior to sample latent variables
            posterior_dist_dendrites = self.posterior_dendrites(
                tf.concat([images, masks_dendrites], axis=-1))
            posterior_dist_spines = self.posterior_spines(
                tf.concat([images, masks_spines], axis=-1))
            
            z_dendrites = posterior_dist_dendrites.sample()
            z_spines = posterior_dist_spines.sample()
        else:
            # During inference, use prior to sample latent variables
            prior_dist = self.prior(tf.shape(images)[0])
            z_dendrites = prior_dist.sample()
            z_spines = prior_dist.sample()
        
        # Reshape latent variables
        z_dendrites = tf.reshape(z_dendrites, [-1, 1, 1, self.latent_dim])
        z_spines = tf.reshape(z_spines, [-1, 1, 1, self.latent_dim])
        
        # Tile to match encoder output spatial dimensions
        z_dendrites = tf.tile(z_dendrites, 
            [1, tf.shape(encoded)[1], tf.shape(encoded)[2], 1])
        z_spines = tf.tile(z_spines, 
            [1, tf.shape(encoded)[1], tf.shape(encoded)[2], 1])
        
        # Concatenate latent variables with encoded features
        encoded_dendrites = tf.concat([encoded, z_dendrites], axis=-1)
        encoded_spines = tf.concat([encoded, z_spines], axis=-1)
        
        # Decode
        output_dendrites = self.decoder_dendrites(encoded_dendrites, skips)
        output_spines = self.decoder_spines(encoded_spines, skips)
        
        if training:
            return (output_dendrites, output_spines), \
                   (posterior_dist_dendrites, posterior_dist_spines)
        return output_dendrites, output_spines
    
    def sample(self, images, num_samples=1):
        dendrite_samples = []
        spine_samples = []
        
        for _ in range(num_samples):
            dendrites, spines = self(images, training=False)
            dendrite_samples.append(dendrites)
            spine_samples.append(spines)
            
        return tf.stack(dendrite_samples, axis=1), tf.stack(spine_samples, axis=1)

