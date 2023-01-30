# Matrix operations
import numpy as np
# Loading data
import flammkuchen as fl
# from keras.utils import Sequence
from tensorflow.keras.utils import Sequence
# Image manipulation
import cv2
# Image augmentations
import albumentations as A
# Shuffling images
import random

class DataGeneratorStream(Sequence):
    def __init__(self, fn, batch_size, samples_per_epoch=50000, size=(1, 128, 128), target_resolution=None, augment=True, 
        shuffle=True, seed=42, normalize=[-1, 1], min_content=0.
    ):
        """Data Generator that streams data dynamically for training DeepD3.

        Args:
            fn (str): The path to the training data file
            batch_size (int): Batch size for training deep neural networks
            samples_per_epoch (int, optional): Samples used in each epoch. Defaults to 50000.
            size (tuple, optional): Shape of a single sample. Defaults to (1, 128, 128).
            target_resolution (float, optional): Target resolution in microns. Defaults to None.
            augment (bool, optional): Enables augmenting the data. Defaults to True.
            shuffle (bool, optional): Enabled shuffling the data. Defaults to True.
            seed (int, optional): Creates pseudorandom numbers for shuffling. Defaults to 42.
            normalize (list, optional): Values range when normalizing data. Defaults to [-1, 1].
            min_content (float, optional): Minimum content in image (annotated dendrite or spine), not considered if 0. Defaults to 0.
        """
        
        # Save settings            
        self.batch_size = batch_size
        self.augment = augment
        self.fn = fn
        self.shuffle = shuffle
        self.aug = self._get_augmenter()
        self.seed = seed
        self.normalize = normalize
        self.samples_per_epoch = samples_per_epoch
        self.size = size
        self.target_resolution = target_resolution
        self.min_content = min_content

        self.d = fl.load(self.fn)
        self.data = self.d['data']
        self.meta = self.d['meta']

        # Seed randomness
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.samples_per_epoch // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data
        
        Parameters
        ----------
        index : int
            batch index in image/label id list
        
        Returns
        -------
        tuple
            Contains two numpy arrays,
            each of shape (batch_size, height, width, 1). 
        """
        X = []
        Y0 = []
        Y1 = []
        eps = 1e-5

        if self.shuffle is False:
            np.random.seed(index)

        # Create all pairs in a given batch
        for i in range(self.batch_size):
            # Retrieve a single sample pair
            image, dendrite, spines = self.getSample()

            # Augmenting the data
            if self.augment:
                augmented = self.aug(image=image, 
                    mask1=dendrite.astype(np.uint8), 
                    mask2=spines.astype(np.uint8)) #augment image
                
                image = augmented['image']
                dendrite = augmented['mask1']
                spines = augmented['mask2']

            # Min/max scaling
            image = (image.astype(np.float32) - image.min()) / (image.max() - image.min() + eps)
            # Shifting and scaling
            image = image * (self.normalize[1]-self.normalize[0]) + self.normalize[0]

            X.append(image)
            Y0.append(dendrite.astype(np.float32) / (dendrite.max() + eps))
            Y1.append(spines.astype(np.float32) / (spines.max() + eps)) # to ensure binary targets

        return np.asarray(X, dtype=np.float32)[..., None], (np.asarray(Y0, dtype=np.float32)[..., None],
                np.asarray(Y1, dtype=np.float32)[..., None])


    def _get_augmenter(self):
        """Defines used augmentations"""
        aug = A.Compose([
            A.RandomBrightnessContrast(p=0.25),    
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.2),
            A.GaussNoise(p=0.5)], p=1,
            additional_targets={
                'mask1': 'mask',
                'mask2': 'mask'
            })
        return aug

    def getSample(self, squeeze=True):
        """Get a sample from the provided data

        Args:
            squeeze (bool, optional): if plane is 2D, skip 3D. Defaults to True.

        Returns:
            list(np.ndarray, np.ndarray, np.ndarray): stack image with respective labels
        """
        while True:
            r = self._getSample(squeeze)
            
            # If sample was successfully generated
            # and we don't care about the content
            if r is not None and self.min_content == 0:
                return r

            # If sample was successfully generated
            # and we do care about the content
            elif r is not None:
                # In either or both annotation should be at least `min_content` pixels
                # that are being labelled.
                if (r[1]).sum() > self.min_content or (r[2]).sum() > self.min_content:
                    return r
                else:
                    continue

            else:
                continue

    def _getSample(self, squeeze=True):
        """Retrieves a sample

        Args:
            squeeze (bool, optional): Squeezes return shape. Defaults to True.

        Returns:
            tuple: Tuple of stack (X), dendrite (Y0) and spines (Y1)
        """
        # Adjust for 2 images
        if len(self.size) == 2:
            size = (1,) + self.size

        else:
            size = self.size
        
        # sample random stack
        r_stack = np.random.choice(len(self.meta))
        
        target_h = size[1]
        target_w = size[2]


        if self.target_resolution is None:
            # Keep everything as is
            scaling = 1
            h = target_h
            w = target_w

        else:
            # Computing scaling factor
            scaling = self.target_resolution / self.meta.iloc[r_stack].Resolution_XY
            
            # Compute the height and width and random offsets
            h = round(scaling * target_h)
            w = round(scaling * target_w)
        
        # Correct for stack dimensions
        if self.meta.iloc[r_stack].Width-w == 0:
            x = 0
            
        elif self.meta.iloc[r_stack].Width-w < 0:
            return
        
        else:
            x = np.random.choice(self.meta.iloc[r_stack].Width-w)

        # Correct for stack dimensions            
        if self.meta.iloc[r_stack].Height-h == 0:
            y = 0
            
        elif self.meta.iloc[r_stack].Height-h < 0:
            return
        
        else:
            y = np.random.choice(self.meta.iloc[r_stack].Height-h)
            
        ## Select random plane + range
        r_plane = np.random.choice(self.meta.iloc[r_stack].Depth-size[0]+1)
        
        z_begin = r_plane
        z_end   = r_plane+size[0]
        
        
        # Scale if neccessary to the correct dimensions
        tmp_stack = self.data['stacks'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]
        tmp_dendrites = self.data['dendrites'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]
        tmp_spines = self.data['spines'][f'x{r_stack}'][z_begin:z_end, y:y+h, x:x+w]

        # Data needs to be rescaled
        if scaling != 1:
            return_stack = []
            return_dendrites = []
            return_spines = []
        
            # Do this for each plane
            # and ensure that OpenCV is happy
            for i in range(tmp_stack.shape[0]):
                return_stack.append(cv2.resize(tmp_stack[i], (target_h, target_w)))
                return_dendrites.append(cv2.resize(tmp_dendrites[i].astype(np.uint8), (target_h, target_w)).astype(bool))
                return_spines.append(cv2.resize(tmp_spines[i].astype(np.uint8), (target_h, target_w)).astype(bool))
                
            return_stack = np.asarray(return_stack)
            return_dendrites = np.asarray(return_dendrites)
            return_spines = np.asarray(return_spines)
            
        else:
            return_stack = tmp_stack
            return_dendrites = tmp_dendrites
            return_spines = tmp_spines
                
        if squeeze:
            # Return sample
            return return_stack.squeeze(), return_dendrites.squeeze(), return_spines.squeeze()
        
        else:
            return return_stack, return_dendrites, return_spines