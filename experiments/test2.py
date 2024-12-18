import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
sm.set_framework("tf.keras")

# Plotting
import matplotlib.pyplot as plt
import wandb

# DeepD3 
from model_in import DeepD3_Model
from datagen import DataGeneratorStream
from helper import MetricsPlotter

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

# Create a naive DeepD3 model with a given base filter count (e.g. 32)
m = DeepD3_Model(filters=32)

# Set appropriate training settings
m.compile(Adam(learning_rate=0.0005), # optimizer, good default setting, can be tuned 
          [sm.losses.dice_loss, "mse"], # Dice loss for dendrite, MSE for spines
          metrics=[sm.metrics.iou_score, sm.metrics.iou_score]) # Metrics for monitoring progress

m.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

def schedule(epoch, lr):
    if epoch < 15:
        return lr
    
    else:
        return float(lr * tf.math.exp(-0.1))

metric_save="metrics_2"
os.makedirs(metric_save,exist_ok=True)

EPOCHS = 30
LEARNING_RATE= 0.0005
run = wandb.init(
        project="Thesis",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS
        })

config = wandb.config
m = DeepD3_Model(filters=32)
m.compile(
    optimizer=Adam(learning_rate=config.learning_rate),
    loss=[sm.losses.dice_loss, "mse"],
    metrics=[sm.metrics.iou_score, sm.metrics.iou_score]
)

# Callbacks
mc = ModelCheckpoint("Experiment_IN.weights.h5", save_best_only=True,save_weights_only=True)
csv = CSVLogger("Experiment_IN.csv")
lrs = LearningRateScheduler(schedule)
metrics_plotter = MetricsPlotter(metric_save)

# Train the model
h = m.fit(
    dg_training,
    batch_size=32,
    epochs=config.epochs,
    validation_data=dg_validation,
    callbacks=[mc, csv, lrs, metrics_plotter]
)

# Save model weights
m.save("Experiment_IN.h5",save_format='h5')

wandb.finish()