import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
sm.set_framework("tf.keras")

# Plotting
import matplotlib.pyplot as plt
import wandb

from datagen import DataGeneratorStream
import numpy as np

TRAINING_DATA_PATH = r"../DeepD3_Training.d3set"
VALIDATION_DATA_PATH = r"../DeepD3_Validation.d3set"

dg_training = DataGeneratorStream(TRAINING_DATA_PATH, 
                                  batch_size=32, # Data processed at once, depends on your GPU
                                  target_resolution=0.094, # fixed to 94 nm, can be None for mixed resolution training
                                  min_content=50) # images need to have at least 50 segmented px

dg_validation = DataGeneratorStream(VALIDATION_DATA_PATH, 
                                    batch_size=32, 
                                    target_resolution=0.094,
                                    min_content=50, 
                                    augment=False,
                                    shuffle=False)

@tf.function
def prob_unet_dual_loss(y_true, y_pred, posterior_dists, prior_dist, 
                        reconstruction_weight=1.0, kl_weight=1.0):
    """
    Custom loss function for dual-output Probabilistic U-Net
    
    Args:
        y_true: Tuple of (dendrite_masks, spine_masks)
        y_pred: Tuple of (dendrite_pred, spine_pred)
        posterior_dists: Tuple of (posterior_dendrites, posterior_spines)
        prior_dist: Prior distribution
        reconstruction_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence loss
    """
    dendrite_true, spine_true = y_true
    dendrite_pred, spine_pred = y_pred
    posterior_dendrites, posterior_spines = posterior_dists
    
    # Reconstruction losses (Binary Cross Entropy)
    recon_loss_dendrites = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(dendrite_true, dendrite_pred))
    recon_loss_spines = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(spine_true, spine_pred))
    
    # KL divergences
    kl_div_dendrites = tf.reduce_mean(
        tfd.kl_divergence(posterior_dendrites, prior_dist))
    kl_div_spines = tf.reduce_mean(
        tfd.kl_divergence(posterior_spines, prior_dist))
    
    # Combine losses
    total_recon_loss = (recon_loss_dendrites + recon_loss_spines) / 2
    total_kl_loss = (kl_div_dendrites + kl_div_spines) / 2
    
    return reconstruction_weight * total_recon_loss + kl_weight * total_kl_loss

class DendriticProbUNetTrainer:
    def __init__(self, model, train_generator, val_generator=None, 
                 learning_rate=1e-4, reconstruction_weight=1.0, kl_weight=1.0):
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.optimizer = Adam(learning_rate)
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        
    @tf.function
    def train_step(self, images, masks):
        with tf.GradientTape() as tape:
            # Forward pass
            predictions, posterior_dists = self.model([images, masks], training=True)
            prior_dist = self.model.prior(tf.shape(images)[0])
            
            # Calculate loss
            loss = prob_unet_dual_loss(
                masks, predictions, posterior_dists, prior_dist,
                self.reconstruction_weight, self.kl_weight
            )
            
        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, predictions
    
    def calculate_dice_score(self, y_true, y_pred, smooth=1e-5):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return (2. * intersection + smooth) / (union + smooth)
    
    def train(self, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            train_loss = []
            train_dice_dendrites = []
            train_dice_spines = []
            
            for i in range(len(self.train_generator)):
                # Get batch of training data
                images, (masks_dendrites, masks_spines) = self.train_generator[i]
                
                # Training step
                loss, (pred_dendrites, pred_spines) = self.train_step(
                    images, (masks_dendrites, masks_spines))
                
                # Calculate Dice scores
                dice_dendrites = self.calculate_dice_score(
                    masks_dendrites, pred_dendrites)
                dice_spines = self.calculate_dice_score(
                    masks_spines, pred_spines)
                
                train_loss.append(loss.numpy())
                train_dice_dendrites.append(dice_dendrites.numpy())
                train_dice_spines.append(dice_spines.numpy())
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(
                        f"Batch {i+1}/{len(self.train_generator)} - "
                        f"Loss: {np.mean(train_loss):.4f} - "
                        f"Dendrite Dice: {np.mean(train_dice_dendrites):.4f} - "
                        f"Spine Dice: {np.mean(train_dice_spines):.4f}"
                    )
            
            # Validation phase
            if self.val_generator is not None:
                val_dice_dendrites = []
                val_dice_spines = []
                
                for i in range(len(self.val_generator)):
                    images, (masks_dendrites, masks_spines) = self.val_generator[i]
                    
                    # Generate multiple samples
                    dendrite_samples, spine_samples = self.model.sample(
                        images, num_samples=5)
                    
                    # Calculate mean predictions
                    mean_dendrites = tf.reduce_mean(dendrite_samples, axis=1)
                    mean_spines = tf.reduce_mean(spine_samples, axis=1)
                    
                    # Calculate dice scores
                    dice_dendrites = self.calculate_dice_score(
                        masks_dendrites, mean_dendrites)
                    dice_spines = self.calculate_dice_score(
                        masks_spines, mean_spines)
                    
                    val_dice_dendrites.append(dice_dendrites.numpy())
                    val_dice_spines.append(dice_spines.numpy())
                
                print(f"\nEpoch {epoch+1} Results:")
                print(f"Training Loss: {np.mean(train_loss):.4f}")
                print(f"Training Dendrite Dice: {np.mean(train_dice_dendrites):.4f}")
                print(f"Training Spine Dice: {np.mean(train_dice_spines):.4f}")
                print(f"Validation Dendrite Dice: {np.mean(val_dice_dendrites):.4f}")
                print(f"Validation Spine Dice: {np.mean(val_dice_spines):.4f}")

# Example usage:
if __name__ == "__main__":
    # Initialize model with appropriate input shape
    input_shape = (128, 128, 1)  # Adjust based on your data
    model = DualOutputProbabilisticUNet(input_shape, filters=32, latent_dim=6)
    
    # Initialize trainer
    trainer = DendriticProbUNetTrainer(
        model, 
        train_generator, 
        val_generator,
        learning_rate=1e-4,
        reconstruction_weight=1.0,
        kl_weight=0.1  # Adjust based on your needs
    )
    
    # Start training
    trainer.train(epochs=1)