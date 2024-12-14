import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
sm.set_framework("tf.keras")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard
import wandb

class MetricsPlotter(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.metrics = {}

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for metric, value in logs.items():
            if metric not in self.metrics:
                self.metrics[metric] = []
            self.metrics[metric].append(value)
        
        wandb.log(logs,step=epoch)
        
        for metric, values in self.metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(values) + 1), values)
            plt.title(f'{metric} vs Epochs')
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.savefig(f'{self.save_dir}/{metric}_vs_epochs.png')
            plt.close()

        plt.figure(figsize=(12, 8))
        for metric, values in self.metrics.items():
            plt.plot(range(1, len(values) + 1), values, label=metric)
        plt.title('All Metrics vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.savefig(f'{self.save_dir}/all_metrics_vs_epochs.png')
        plt.close()