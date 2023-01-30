import numpy as np
from roifile import ImagejRoi, roiwrite
from PyQt5.QtCore import QObject, pyqtSignal
import pandas as pd 
import imageio as io
import os
from pathlib import Path

class ExportFolder(QObject):
    zSignal = pyqtSignal(int, int)

    def __init__(self, rois):
        """Class to export ROIs to a folder

        Args:
            rois (dict): dictionary of ROIs, keys are z-plane, value is list with ROIs in given plane.
        """
        super().__init__()
        self.rois = rois

    def export(self, fn, folder):
        """Export ROIs to folder

        Args:
            fn (str): file name
            folder (str): target folder
        """
        basename = Path(fn).stem

        # Folder for filename
        target = os.path.join(folder, basename)
        
        if not os.path.exists(target):
            os.mkdir(target)
        
        for k, v in self.rois.items():
            subtarget = os.path.join(target, str(k))
            
            # Folder for Z
            if not os.path.exists(subtarget):
                os.mkdir(subtarget)
                
            for i in v:
                # Each ROI in this Z
                df = pd.DataFrame(i['contour'])
                df.columns = 'y', 'x'
                df.to_csv(os.path.join(subtarget, str(i['ROI_id'])+".csv"), index=False)

            # Tell GUI that there's some progress
            self.zSignal.emit(int(z), len(self.rois.keys()))


class ExportImageJ(QObject):
    zSignal = pyqtSignal(int, int)

    def __init__(self, rois):
        """Class to export ROIs to ImageJ ROI zip file

        Args:
            rois (dict): dictionary of ROIs, keys are z-plane, value is list with ROIs in given plane.
        """
        super().__init__()
        self.rois = rois

    def export(self, fn):
        """Export ROIs to ImageJ ROI zip file

        Args:
            fn (str): path to zip file
        """

        # Iterate over z-planes
        for z, v in self.rois.items():

            # Iterate over available ROIs
            for i in v:
                # Create ImageJ ROI from contour,
                # change x and y for ImageJ logic
                r = ImagejRoi.frompoints(i['contour'][:, ::-1])

                # ImageJ z starts with 1, correct for it
                r.z_position = z+1

                # Write to ROI to zipfile
                roiwrite(fn, r)

            # Tell GUI that there's some progress
            self.zSignal.emit(int(z), len(self.rois.keys()))

class ExportCentroids(QObject):
    def __init__(self, roi_centroids) -> None:
        """Class to export ROI centroids to file

        Args:
            roi_centroids (dict): ROI centroids
        """
        super().__init__()
        self.roi_centroids = roi_centroids

    def export(self, fn):
        """Exports ROIs to file

        Args:
            fn (str): target filename and location
        """
        tmp = []

        for i in self.roi_centroids:
            tmp.append(dict(Pos=i[0], Y=i[1], X=i[2]))

        pd.DataFrame(tmp).to_csv(fn)


class ExportPredictions(QObject):
    def __init__(self, pred_spines, pred_dendrites):
        """Class to export ROIs to a folder

        Args:
            rois (dict): dictionary of ROIs, keys are z-plane, value is list with ROIs in given plane.
        """
        super().__init__()
        self.pred_spines = pred_spines
        self.pred_dendrites = pred_dendrites

    def export(self, fn, folder):
        """Export predictions as tif files

        Args:
            fn (str): file name
            folder (str): target folder
        """
        basename = Path(fn).stem

        # Folder for filename
        target_spines    = os.path.join(folder, basename+"_spines.tif")
        target_dendrites = os.path.join(folder, basename+"_dendrites.tif")
        
        try:
            io.mimwrite(target_spines, (self.pred_spines * 255).astype(np.uint8))
            io.mimwrite(target_dendrites, (self.pred_dendrites * 255).astype(np.uint8))

            return True, target_spines+"\n"+target_dendrites

        except Exception as e:
            return False, e


class ExportROIMap(QObject):
    def __init__(self, roi_map, binarize=False):
        """Class to export ROIs to a folder

        Args:
            rois (dict): dictionary of ROIs, keys are z-plane, value is list with ROIs in given plane.
        """
        super().__init__()
        self.roi_map = roi_map
        self.binarize = binarize

    def export(self, fn):
        """Export predictions as tif files

        Args:
            fn (str): file name
            folder (str): target folder
        """
        try:
            if not self.binarize:
                N = self.roi_map.max()

                if N < 2**8:
                    dtype = np.uint8
                if N < 2**16:
                    dtype = np.uint16
                else:
                    dtype = np.int32

                exp = self.roi_map.astype(dtype)

            else: 
                exp = (self.roi_map > 0).astype(np.uint8) * 255

            io.mimwrite(fn, exp)

            return True, ""

        except Exception as e:
            return False, e
