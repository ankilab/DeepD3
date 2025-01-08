import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculates the Dice coefficient."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss (complement of the Dice coefficient)."""
    return 1. - dice_coefficient(y_true, y_pred)

def kl_divergence_loss(mu, log_variance):
    """KL divergence loss."""
    kl_loss = -0.5 * tf.reduce_sum(1 + log_variance - tf.square(mu) - tf.exp(log_variance))
    return kl_loss

def combined_loss(y_true, y_pred, mu, log_variance, alpha=0.1):
    """Combines Dice loss and KL divergence loss."""
    dice = dice_loss(y_true, y_pred)
    kl = kl_divergence_loss(mu, log_variance)
    return dice + alpha * kl