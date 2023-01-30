import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import imageio as io
import flammkuchen as fl
import os
import cv2
from skimage.measure import find_contours, moments
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.draw import disk
from tqdm import tqdm
import cc3d
from numba import njit
from deepd3.core.dendrite import sphere
from scipy.ndimage import grey_closing, binary_dilation, label as labelImage, distance_transform_edt

@njit
def centroid3D(im):
    """Computes centroid from a 3D binary image

    Args:
        im (numpy.ndarray): binary image

    Returns:
        tuple: centroid coordinates z,y,x
    """
    z = 0
    y = 0
    x = 0
    n = 0
    
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            for k in range(im.shape[2]):
                if im[i,j,k]:
                    z += i
                    y += j
                    x += k
                    n += 1
                 
    if n == 0:
        return 0, 0, 0
    else:
        return z/n, y/n, x/n

@njit(cache=True)
def centroids3D_from_labels(labels):
    """Computes the centroid for each label in an 3D stack containing image labels.
    0 is background, 1...N are foreground labels.
    This function uses image moments to compute the centroid.

    Args:
        labels (numpy.ndarray): ROI labeled image (0...N)

    Returns:
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray): 
            Returns first-order moments, zero-order moments and covered planes
    """
    N = np.max(labels)
    # First-order moments for z, y, and x
    cs = np.zeros((N+1, 3), dtype=np.float32)
    # Zero-order moment (area/mass of label)
    px = np.zeros(N+1, dtype=np.float32)
    # Coverage of planes (ROIs x z-planes)
    planes = np.zeros((N+1, labels.shape[0]), dtype=np.int32)
    
    # Fast for-loops using numba
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                l = labels[i,j,k]
                cs[l, 0] += i # M 1,0,0
                cs[l, 1] += j # M 0,1,0
                cs[l, 2] += k # M 0,0,1
                px[l] += 1    # M 0,0,0
                planes[l,i] = 1 # plane coverage
                 
    return cs, px, planes

@njit(cache=True)
def minMaxProbability(labels, prediction):
    """Computes the minimum and maximum probabilty of a prediction map given a label map

    Args:
        labels (numpy.ndarray): labels
        prediction (numpy.ndarray): prediction map with probabilities 0 ... 1

    Returns:
        numpy.ndarray: for each label the minimum and maximum probability
    """
    probs = np.ones((labels.max(), 2))
    probs[:, 1] = 0

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                l = labels[i,j,k]
                p = prediction[i,j,k]

                if probs[l, 0] > p:
                    probs[l, 0] = p

                if probs[l, 1] < p:
                    probs[l, 1] = p

    return probs


@njit(cache=True)
def cleanLabels(labels, rois_to_delete):
    """Cleans labels from label stack. Set labels in rois_to_delete to background.

    Args:
        labels ([type]): [description]
        rois_to_delete ([type]): [description]

    Returns:
        [type]: [description]
    """
    clean_labels = np.zeros(labels.shape, dtype=np.int32)
    
    # Iterate through stack
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                l = labels[i,j,k]

                # If it is background label, go on
                if l == 0:
                    continue
                
                # Not background label, check if label should remain
                if (rois_to_delete==l).sum() > 0:
                    clean_labels[i,j,k] = 0

                else:
                    clean_labels[i,j,k] = l
                 
    return clean_labels

@njit(cache=True)
def reid(labels):
    """Relabel an existing label map to ensure continuous label ids

    Args:
        labels (numpy.ndarray): original label map

    Returns:
        numpy.ndarray: re-computed label map
    """
    ls = np.unique(labels)
    
    new_labels = np.zeros(labels.shape, dtype=np.int32)
    
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                l = labels[i,j,k]
                
                if l == 0:
                    continue
                    
                else:
                    new_labels[i,j,k] = np.argmax(ls==l)
                
    return new_labels

@njit(cache=True)
def getROIsizes(labels):
    """Get the ROI size for each label with one single stack pass

    Args:
        labels (numpy.ndarray): label map

    Returns:
        numpy.ndarray: the size of each ROI area
    """
    roi_sizes = np.zeros(np.max(labels), dtype=np.uint16)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                l = labels[i,j,k]
                roi_sizes[l] += 1

    return roi_sizes

##########################
# 3D connected components 
##########################

# @njit(cache=True)
def _get_sorted_seeds(stack, threshold=0.8):
    """Sort seeds according to their highest prediction value

    Args:
        stack (numpy ndarray): The stack with the predictions
        threshold (float, optional): The threshold for being a seed pixel. Defaults to 0.8.

    Returns:
        numpy.ndarray: seed coordinates sorted by prediction value
    """
    coords = np.nonzero(stack>=threshold)
    intensities = stack[coords]
    # Highest peak first
    idx_maxsort = np.argsort(-intensities)
    coords = np.transpose(coords)[idx_maxsort]
    return coords

@njit(cache=True)
def _neighbours(x,y,z):
    """Generates 26-connected neighbours

    Args:
        x (int): x-value
        y (int): y-value
        z (int): z-value

    Returns:
        list: neighbour indices of a given point (x,y,z)
    """
    look = []
    
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            for k in range(z-1, z+2):
                if not (i == x and j == y and k == z):
                    look.append((i,j,k))
                    
    return look


@njit
def _distance_to_seed(seed, pos, delta_xy = 1, delta_z = 1):
    """Computes the euclidean distance between seed pixel and current position `pos`

    Args:
        seed (tuple): seed pixel coordinates (x,y,z)
        pos (tuple): current position coordinates (x,y,z)

    Returns:
        float: euclidean distance between seed and current position 
    """
    a = (seed[0] * delta_xy - pos[0] * delta_xy)**2
    b = (seed[1] * delta_xy - pos[1] * delta_xy)**2
    c = (seed[2] * delta_z  - pos[2] * delta_z)**2

    return np.sqrt(a+b+c)

@njit
def connected_components_3d(prediction, seeds, delta, threshold, distance, dimensions):
    """Computes connected components in 3D using various constraints.
    Each ROI is grown from a seed pixel. From there, in a 26-neighbour fashion more
    pixels are added iteratively. Each additional pixel needs to fulfill the following requirements:

    * The new pixel's intensity needs to be in a given range relative to the seed intensity (`delta`)
    * The new pixel's intensity needs to be above a given `threshold`
    * The new pixel's position needs to be in the vicinity (`distance`) of the seed pixel

    Each pixel can only be assigned to one ROI once.

    Args:
        prediction (numpy.ndarray): prediction from deep neural network
        seeds (numpy.ndarray): seed pixels
        delta (float): difference to seed pixel intensity
        threshold (float): threshold for pixel intensity
        distance (int or float): maximum euclidean distance in microns to seed pixel
        dimensions (dict(float, float)): xy and z pitch in microns

    Returns:
        tuple(labels, N): the labelled stack and the number of found ROIs
    """
    # Initialize everything as background
    im = np.zeros(prediction.shape, dtype=np.uint16) 
    L = 1 # Start with label 1

    delta_xy = dimensions[0]
    delta_z  = dimensions[1]

    # Iterate through seed pixels
    for i in range(seeds.shape[0]):     
        # Retrieve location and seed intensity   
        x0, y0, z0 = seeds[i]
        t = prediction[x0, y0, z0]
        
        # Seed pixel has been assigned to a label already, skip
        if im[x0, y0, z0]:
            continue
        
        # Start with the floodfilling
        neighbours = [(x0, y0, z0)]
        
        while len(neighbours):
            # Look at next pixel
            x, y, z = neighbours.pop()
            
            # Current pixel not in stack
            if x >= im.shape[0] or x < 0:
                continue
                
            if y >= im.shape[1] or y < 0:
                continue
                
            if z >= im.shape[2] or z < 0:
                continue
                
            # Intensity at given point
            p0 = prediction[x, y, z]
            
            # A good pixel should be
            # - similar to the seed px (delta)
            # - intensity above a given threshold 
            # - in label image it is still a background px
            # - distance to seed is lower than distance
            if abs(p0 - t) <= delta * t and \
                 p0 > threshold and \
                 im[x, y, z] == 0 and \
                 _distance_to_seed((x0, y0, z0), (x,y,z), delta_xy, delta_z) < distance:

                # Assign pixel current label
                im[x,y,z] = L
                
                # Look at neighbours
                neighbours.extend(_neighbours(x,y,z))

        # Finished with this label 
        L += 1
                
    return im, L-1


class Stack(QObject):
    tileSignal = pyqtSignal(int, int)

    def __init__(self, fn, pred_fn=None, dimensions=dict(xy=0.094, z=0.5)):
        """Stack

        Args:
            fn (str): Path to file to be openend via imagio
            pred_fn (str, optional): Path to prediction files. Defaults to None.
            dimensions (dict, optional): XY and Z dimensions. Defaults to dict(xy=0.094, z=0.5).
        """
        super().__init__()
        self.stack = np.asarray(io.mimread(fn, memtest=False))

        # If stack is only an image, create dummy dimension
        if len(self.stack.shape) == 2:
            self.stack = self.stack[None]

        # Prediction and preview should be pre-allocated
        self.prediction = np.zeros(self.stack.shape+(3,), dtype=np.float32) 
        self.preview    = np.zeros(self.stack.shape+(3,), dtype=np.float32) 
        self.segmented = False
        self.dimensions = dimensions

        # If predictions are already existing
        if pred_fn is not None:
            if os.path.exists(pred_fn):
                pred = fl.load(pred_fn)

                dendrite_key = 'dendrites' if 'dendrites' in pred.keys() else 'dendrite'

                # Reload and assign dendrite and spine keys
                self.prediction[..., 0] = pred[dendrite_key]
                self.prediction[..., 1] = pred['spines']
                self.prediction[..., 2] = pred[dendrite_key]
                self.segmented = True

    def __getitem__(self, sl):
        """Get item slice

        Args:
            sl (int): stack index

        Returns:
            numpy.array: image
        """
        return self.stack[sl]

    def cleanSpines(self, dendrite_threshold=0.7, dendrite_dilation_iterations=12, preview=False):
        """Cleaning spines in 2D

        Args:
            dendrite_threshold (float, optional): Dendrite threshold for segmentation. Defaults to 0.7.
            dendrite_dilation_iterations (int, optional): Iterations to enlarge dendrite. Defaults to 12.
            preview (bool, optional): Enable preview option (not overwriting predictions). Defaults to False.

        Returns:
            numpy.ndarray: cleaned spines stack
        """
        if preview:
            d = self.preview[..., 0].copy()
            s = self.preview[..., 1].copy()
        else:
            d = self.prediction[..., 0].copy()
            s = self.prediction[..., 1].copy()

        bd = binary_dilation(np.asarray(d) > dendrite_threshold, 
            iterations=dendrite_dilation_iterations)

        s[~bd] = 0

        return s

    def cleanDendrite3D(self, dendrite_threshold=0.7, min_dendrite_size=100, preview=False):
        """Cleaning dendrites in 3D

        Args:
            dendrite_threshold (float, optional): Dendrite semantic segmentation threshold. Defaults to 0.7.
            min_dendrite_size (int, optional): Minimum dendrite size in px in 3D. Defaults to 100.
            preview (bool, optional): Enable preview option. Defaults to False.

        Returns:
            numpy.ndarray: Cleaned dendrite
        """
        
        if preview:
            d = self.preview[..., 0].copy()

        else:
            d = self.prediction[..., 0].copy()
        
        # Clean noisy px
        d[d < dendrite_threshold] = 0

        # Create labels for all dendritic elements
        labels, N = cc3d.connected_components(d > dendrite_threshold, return_N=True)

        # Compute ROI sizes
        roi_sizes = getROIsizes(labels)

        # Remove dendrite segments 
        new_labels = cleanLabels(labels, np.where(roi_sizes < min_dendrite_size)[0])

        # clean data
        d_clean = d * (new_labels > 0).astype(np.float32)

        return d_clean

    def closing(self, iterations=1, preview=False):
        """Closing operation on dendrite prediction map

        Args:
            iterations (int, optional): Iterations of closing operation. Defaults to 1.
            preview (bool, optional): Enables preview mode. Defaults to False.

        Returns:
            numpy.ndarray: cleaned dendrite map
        """
        if preview:
            d = self.preview[..., 0].copy()

        else:
            d = self.prediction[..., 0].copy()

        d_clean = d
        
        for _ in range(iterations):
            d_clean = grey_closing(d_clean, size=(3,3,3))

        return d_clean

    def cleanDendrite(self, dendrite_threshold=0.7, min_dendrite_size=100):
        """Cleaning dendrite 

        Args:
            dendrite_threshold (float, optional): Dendrite probability threshold. Defaults to 0.7.
            min_dendrite_size (int, optional): Minimum dendrite size. Defaults to 100.

        Returns:
            numpy.ndarray: Cleaned dendrite prediction map
        """
        clean = np.zeros_like(self.prediction[..., 0])

        # Iterate across planes...
        for z in tqdm(range(self.prediction.shape[0])):
            # Retrieve dendrite prediction
            d = self.prediction[z, ..., 0] .copy()

            # Threshold dendrite
            d_thresholded = (d > dendrite_threshold).astype(np.uint8) * 255
            d_clean = np.zeros_like(d)

            # Find elements
            no, labels, _, _ = cv2.connectedComponentsWithStats(d_thresholded)

            for l in range(1, no):
                if (labels==l).sum() > min_dendrite_size:
                    d_clean[labels==l] = d[labels==l]
                
            clean[z] = d_clean

            self.tileSignal.emit(z, self.prediction.shape[0])

        return clean
            
    def predictInset(self, model_fn, tile_size=128, inset_size=96, pad_op=np.mean, zmin=None, zmax=None, clean_dendrite=True, dendrite_threshold=0.7):
        """Predict inset

        Args:
            model_fn (str): path to Tensorflow/Keras model
            tile_size (int, optional): Size of full tile. Defaults to 128.
            inset_size (int, optional): Size of tile inset (probability map to be kept). Defaults to 96.
            pad_op (_type_, optional): Padding operation. Defaults to np.mean.
            zmin (_type_, optional): Z-index minimum. Defaults to None.
            zmax (_type_, optional): Z-index maxmimum. Defaults to None.
            clean_dendrite (bool, optional): Cleaning dendrite. Defaults to True.
            dendrite_threshold (float, optional): Dendrite probability threshold. Defaults to 0.7.

        Returns:
            bool: operation was successful
        """
        from tensorflow.keras.models import load_model
        model = load_model(model_fn, compile=False)

        # Check for z-range, if nothing is provided, assume the whole stack
        zmin = zmin if zmin else 0
        zmax = zmax if zmax else self.stack.shape[0]

        # Compute rim offset between tile_size and inset_size
        off = (tile_size-inset_size)//2

        # Compute the image size that is needed
        h = int(np.ceil(self.stack.shape[1] / inset_size) * inset_size)
        w = int(np.ceil(self.stack.shape[2] / inset_size) * inset_size)

        # Values for padding
        cv = pad_op(self.stack)

        # Padding
        stack_zp = np.pad(self.stack, # Pad the stack
                          ((0, 0), # no pad in z
                           (off, h-self.stack.shape[1]+off), # pad in y
                           (off, w-self.stack.shape[2]+off)), # pad in x
                          constant_values=cv)

        predictions = np.zeros(stack_zp.shape + (3,), dtype=np.float32)

        steps_y = np.arange(0, h, inset_size).astype(np.int)
        steps_x = np.arange(0, w, inset_size).astype(np.int)

        # Iterate over tiles, y
        for i in tqdm(steps_y):
            # Iterate over tiles, x
            for j in steps_x:
                # Take stack tile (column in z) at the respective tile position
                tile = stack_zp[zmin:zmax, i:i+tile_size, j:j+tile_size] #.copy()
                # Prepare tile for network inference
                tile = (tile.astype(np.float32)-tile.min()) / (tile.max()-tile.min()) * 2 - 1

                # Predict dendrite (pd) and spines (ps), add 1 pseudo-ch
                pd, ps = model.predict(tile[..., None])   

                # Save inset at tile position in prediction stack across z
                if tile_size != inset_size:
                    predictions[zmin:zmax, i+off:i+tile_size-off, j+off:j+tile_size-off, 0] = pd.squeeze()[:, off:-off, off:-off]
                    predictions[zmin:zmax, i+off:i+tile_size-off, j+off:j+tile_size-off, 1] = ps.squeeze()[:, off:-off, off:-off]
                    predictions[zmin:zmax, i+off:i+tile_size-off, j+off:j+tile_size-off, 2] = pd.squeeze()[:, off:-off, off:-off]

                else:
                    predictions[zmin:zmax, i:i+tile_size, j:j+tile_size, 0] = pd.squeeze()
                    predictions[zmin:zmax, i:i+tile_size, j:j+tile_size, 1] = ps.squeeze()
                    predictions[zmin:zmax, i:i+tile_size, j:j+tile_size, 2] = pd.squeeze()

            self.tileSignal.emit(np.argmax(steps_y==i), steps_y.size)
            
        self.prediction = predictions[:, off:off+self.stack.shape[1], off:off+self.stack.shape[2]]

        self.segmented = True
        return True

    def predictWholeImage(self, model_fn):
        """Predict whole image, plane by plane

        Args:
            model_fn (str): path to Tensorflow/Keras model file

        Returns:
            bool: operation was successful
        """
        from tensorflow.keras.models import load_model

        model = load_model(model_fn, compile=False)

        # Iterate over tiles, y
        for z in tqdm(range(self.stack.shape[0])):
            # Iterate over tiles, x
            plane = self.stack[z].copy()
            plane = (plane - plane.min()) / (plane.max()-plane.min()) * 2 - 1

            h, w = plane.shape[0], plane.shape[1]

            # if height or widht is not divisible by 32 (issue with neural networks)
            if h % 32 or w % 32:
                plane = np.pad(plane, # Pad the plane
                          ((0, 32-h%32 if h%32 else 0), # pad in y
                           (0, 32-w%32 if w%32 else 0)), # pad in x
                          mode='reflect') # already normalized

            # Predict dendrite (pd) and spines (ps), add 1 pseudo-ch
            pd, ps = model.predict(plane[None, ..., None])   

            # Dendrite and Spine prediction, crop back
            d = pd.squeeze()[:h, :w]
            s = ps.squeeze()[:h, :w]

            self.prediction[z, ..., 0] = d
            self.prediction[z, ..., 1] = s
            self.prediction[z, ..., 2] = d

            self.tileSignal.emit(z, self.stack.shape[0])

        self.segmented = True
        return True

    def predictFourFold(self, model_fn, tile_size=128, inset_size=96, pad_op=np.mean, zmin=None, zmax=None):
        """Similar to `predictInset` (single tile prediction), but with four-way correction

        Args:
            model_fn (str): path to Tensorflow/Keras model
            tile_size (int, optional): Size of full tile. Defaults to 128.
            inset_size (int, optional): Size of tile inset (probability map to be kept). Defaults to 96.
            pad_op (_type_, optional): Padding operation. Defaults to np.mean.
            zmin (_type_, optional): Z-index minimum. Defaults to None.
            zmax (_type_, optional): Z-index maxmimum. Defaults to None.

        Returns:
            bool: operation was successful
        """
        
        from tensorflow.keras.models import load_model
        model = load_model(model_fn, compile=False)

        # Check for z-range, if nothing is provided, assume the whole stack
        zmin = zmin if zmin else 0
        zmax = zmax if zmax else self.stack.shape[0]

        # Create empty array for z-range and 3 channels (could be two, but then color is easy going)
        off = (tile_size-inset_size)//2

        # Zero pad image to ensure that full stack is analyzed
        cv = pad_op(self.stack)
        stack_zp = np.pad(self.stack, ((0, 0), (tile_size, tile_size), (tile_size, tile_size)), constant_values=cv)

        # all four predictions to be stored
        predictions = np.zeros((4,)+stack_zp.shape + (3,), dtype=np.float32)

        # Predict stack 4 times with different offsets to ensure
        # that the prediction is properly done at the edges
        for it, (start_y, start_x) in enumerate([(0, 0), (tile_size//2, 0), (0, tile_size//2), (tile_size//2, tile_size//2)]):
            steps_y = np.arange(start_y, self.stack.shape[1], inset_size).astype(np.int)
            steps_x = np.arange(start_x, self.stack.shape[2], inset_size).astype(np.int)

            # Iterate over tiles, y
            for i in tqdm(steps_y):
                # Iterate over tiles, x
                for j in steps_x:
                    # Take stack tile (column in z) at the respective tile position
                    tile = stack_zp[zmin:zmax, i:i+tile_size, j:j+tile_size] #.copy()
                    # Prepare tile for network inference
                    tile = (tile.astype(np.float32)-tile.min()) / (tile.max()-tile.min()) * 2 - 1

                    # Predict dendrite (pd) and spines (ps), add 1 pseudo-ch
                    pd, ps = model.predict(tile[..., None])   

                    # Save inset at tile position in prediction stack across z
                    if tile_size != inset_size:
                        predictions[it, zmin:zmax, i+off:i+tile_size-off, j+off:j+tile_size-off, 0] = pd.squeeze()[:, off:-off, off:-off]
                        predictions[it, zmin:zmax, i+off:i+tile_size-off, j+off:j+tile_size-off, 1] = ps.squeeze()[:, off:-off, off:-off]
                        predictions[it, zmin:zmax, i+off:i+tile_size-off, j+off:j+tile_size-off, 2] = pd.squeeze()[:, off:-off, off:-off]

                    else:
                        predictions[it, zmin:zmax, i:i+tile_size, j:j+tile_size, 0] = pd.squeeze()
                        predictions[it, zmin:zmax, i:i+tile_size, j:j+tile_size, 1] = ps.squeeze()
                        predictions[it, zmin:zmax, i:i+tile_size, j:j+tile_size, 2] = pd.squeeze()

                self.tileSignal.emit(np.argmax(steps_y==i), steps_y.size)

        self.prediction = predictions.mean(0)[:, tile_size:-tile_size, tile_size:-tile_size]

        self.segmented = True
        return True


class ROI3D_Creator(QObject):
    zSignal = pyqtSignal(int, int)
    log = pyqtSignal(str)

    def __init__(self, dendrite_prediction, spine_prediction, mode='floodfill', areaThreshold=0.25, 
        peakThreshold=0.8, seedDelta=0.1, distanceToSeed=10, dimensions=dict(xy=0.094, z=0.5)):
        """3D ROI Creator.

        Given the arguments, 3D ROIs are built dynamically from dendrite and spine prediction.

        Args:
            dendrite_prediction (numpy.ndarray): dendrite prediction probability stack
            spine_prediction (numpy.ndarray): spine prediction probability stack
            mode (str, optional): Mode for building 3D rois (floodfill or connected components). Defaults to 'floodfill'.
            areaThreshold (float, optional): Area threshold for floodfilling and connected components. Defaults to 0.25.
            peakThreshold (float, optional): Peak threshold for finding seed points. Defaults to 0.8.
            seedDelta (float, optional): Difference to seed in terms of relative probability. Defaults to 0.1.
            distanceToSeed (int, optional): Distance to seed px in micrometer. Defaults to 10.
            dimensions (dict, optional): Dimensions in xy and z in in micrometer. Defaults to dict(xy=0.094, z=0.5).
        """
        
        super().__init__()
        self.dendrite_prediction = dendrite_prediction
        self.spine_prediction = spine_prediction
        self.mode = mode # floodfill or thresholded
        self.areaThreshold = areaThreshold
        self.peakThreshold = peakThreshold
        self.seedDelta = seedDelta 
        self.distanceToSeed = distanceToSeed
        self.dimensions = dimensions
        self.roi_map = np.zeros_like(spine_prediction, dtype=np.int32)
        self.rois = {}

        self.computeContours = True

    def create(self, minPx, maxPx, minPlanes, applyWatershed=False, dilation=0):
        """Create 3D ROIs

        Args:
            minPx (int): only retain 3D ROIs containing at least `minPx` pixels
            maxPx (int): only retain 3D ROIs containing at most `maxPx` pixels
            minPlanes (int): only retain 3D ROIs spanning at least `minPlanes` planes
            applyWatershed (bool, optional): Apply watershed algorithm to divide ROIs. Defaults to False.
            dilation (int, optional): Dilate dendrite probability map. Defaults to 0.

        Returns:
            int: number of retained ROIs
        """
        ROI_id = 0

        # Find raw labels 
        self.log.emit("Create labels")

        if self.mode == 'floodfill':
            # Find all potential seed pixels that are at least at peakThreshold
            seeds = _get_sorted_seeds(self.spine_prediction, self.peakThreshold)

            # Generate all labels using custom 3D connected components
            labels, N = connected_components_3d(self.spine_prediction,
                seeds,
                self.seedDelta,
                self.areaThreshold,
                self.distanceToSeed,
                (self.dimensions['xy'], self.dimensions['z']))

        else:
            thresholded_im = self.spine_prediction > self.areaThreshold

            if dilation > 0:
                thresholded_im = binary_dilation(thresholded_im, np.ones((3,3,3)), iterations=dilation)

            labels, N = cc3d.connected_components(thresholded_im, return_N=True)

        if applyWatershed:
            D = distance_transform_edt(labels > 0)
            # Seed generation
            localMax = peak_local_max(D, indices=False, min_distance=0, footprint=np.ones((3,3,3)), exclude_border=1)

            markers = labelImage(localMax, structure=np.ones((3,3,3)))[0]

                                        ### data # seeds   # eliminate noise
            labels = watershed(-D, markers, mask=labels > 0)

        self.log.emit("Compute meta data")
        # Compute raw centroids, check size and plane span of ROIs
        cs, px, planes = centroids3D_from_labels(labels)
        planes = planes.sum(1)

        
        self.log.emit("Clean ROIs")
        # Find ROIs that do not match the criteria
        criteria_mismatch = (planes < minPlanes) | (px < minPx) | (px > maxPx)
        rois_to_delete = np.where(criteria_mismatch)[0]

        self.log.emit(f"Removing {len(rois_to_delete)} ROIs...")

        self.log.emit("Clean labels")
        # Clean all labels
        labels = cleanLabels(labels, rois_to_delete)
        labels = reid(labels)

        # Re-compute information
        cs, px, planes = centroids3D_from_labels(labels)
        planes = planes.sum(1)

        # Compute the centroids again after cleaning the labels
        centroids = cs/px[:, None]

        self.roi_centroids = centroids

        if self.computeContours:

            self.log.emit("Compute contours and create plane-wise slices")
            for z in tqdm(range(self.spine_prediction.shape[0])):
                self.rois[z] = []
                labels_plane = labels[z]
                
                # Compute contours...
                for ROI_id in np.unique(labels_plane):
                    if ROI_id == 0:
                        continue

                    c = find_contours(labels_plane==ROI_id)
                    self.rois[z].append({
                        'ROI_id': ROI_id,
                        'contour': c[0],
                        'centroid': np.asarray(centroids[ROI_id])
                    })

                # Talk to progress bar
                self.zSignal.emit(z, self.spine_prediction.shape[0])

        self.roi_map = labels 
        self.log.emit("Done.")
        return np.max(labels)


class ROI2D_Creator(QObject):
    zSignal = pyqtSignal(int, int)

    def __init__(self, dendrite_prediction, spine_prediction, threshold):
        """2D ROI Creator.

        Creates 2D ROIs dependent on dendrite and spine prediction, as well as threshold

        Args:
            dendrite_prediction (_type_): _description_
            spine_prediction (_type_): _description_
            threshold (_type_): _description_
        """
        super().__init__()
        self.dendrite_prediction = dendrite_prediction
        self.spine_prediction = spine_prediction
        self.threshold = threshold
        self.roi_map = np.zeros_like(spine_prediction, dtype=np.int32)
        self.rois = {}

    def create(self, applyWatershed=False, maskSize=3, minDistance=3):
        """Creates 2D ROIs

        Args:
            applyWatershed (bool, optional): Apply Watershed algorithm. Defaults to False.
            maskSize (int, optional): Size of the distance transform mask. Defaults to 3.
            minDistance (int, optional): Minimum distance between ROIs in Watershed algorithm. Defaults to 3.

        Returns:
            int: ROIs found
        """
        ROIs_found = 0
        ROI_id = 0

        for z in tqdm(range(self.spine_prediction.shape[0])):
            self.rois[z] = []

            im = (self.spine_prediction[z] > self.threshold).astype(np.uint8) * 255

            if applyWatershed:
                D = cv2.distanceTransform(im, cv2.DIST_L2, maskSize)
                Ma = peak_local_max(D, indices=False, footprint=np.ones((3,3)), min_distance=minDistance, labels=im)
                foreground_labels = cv2.connectedComponents(Ma.astype(np.uint8)*255)[1]

                labels = watershed(-D, foreground_labels, mask=im)
                
                centroids = []

                uq = np.unique(labels)
                no = len(uq)

                for ix in uq:
                    M = moments(labels==ix)
                    cy, cx = M[1,0]/M[0,0], M[0,1]/M[0,0]
                    centroids.append([cx,cy])

                self.roi_map[z] = labels

            else:
                # find individual ROIs
                no, labels, stats, centroids = cv2.connectedComponentsWithStats(im)
                self.roi_map[z] = labels

            # Compute contours...
            for roi in range(1, no):
                c = find_contours(labels==roi)
                self.rois[z].append({
                    'ROI_id': ROI_id,
                    'contour': c[0],
                    'centroid': np.asarray(centroids[roi])
                })
                ROI_id += 1

            # Talk to progress bar
            self.zSignal.emit(z, self.spine_prediction.shape[0])

            ROIs_found += no-1

        return ROIs_found

    def clean(self, maxD, minS, dendrite_threshold=0.7):
        """Cleanes ROIs

        Args:
            maxD (int): maximum distance to dendrite in px
            minS (int): minimum size of ROIs in px
            dendrite_threshold (float, optional): _description_. Defaults to 0.7.

        Returns:
            tuple: old ROI count, new ROI count
        """
        clean_rois = {}
        clean_roi_map = np.zeros_like(self.roi_map)

        old_rois_count = 0
        new_rois_count = 0

        ### Iterate over z
        for z, rois in tqdm(self.rois.items()):
            clean_rois[z] = []
            ### Iterate over ROIs, check if (centroid +- maxD).sum() > 0
            for i, roi in enumerate(rois):
                old_rois_count += 1
                cur_clean_roi = 0

                # Get ROI coordinates
                x, y = roi['centroid']
                y = int(y)
                x = int(x)
                z = int(z)
                maxD = int(maxD)

                # Retrieve circular area around the ROI center of mass
                rr, cc = disk((y,x), maxD, shape=self.dendrite_prediction.shape[1:3])
                area = self.dendrite_prediction[z, rr, cc]
            
                # Test if any dendritic pixel are inside of this circle
                # and if the ROI size exceeds a given threshold
                dendrite_proximity = (area > dendrite_threshold).sum()
                roi_size = (self.roi_map[z] == i+1).sum()

                if dendrite_proximity and roi_size >= minS:
                    new_rois_count += 1
                    cur_clean_roi += 1
                    clean_rois[z].append(roi)
                    clean_roi_map[z][self.roi_map[z] == i+1] = cur_clean_roi

            # Talk to progress bar
            self.zSignal.emit(int(z), len(self.rois.keys()))

        self.rois = clean_rois
        self.roi_map = clean_roi_map

        return old_rois_count, new_rois_count




