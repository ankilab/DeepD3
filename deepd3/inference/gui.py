import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, \
    QMessageBox, QFileDialog, QGridLayout, QLabel, QPushButton, \
    QProgressBar, QDialog, QTableWidget, QTableWidgetItem, QHeaderView, \
    QLineEdit, QAction, QGraphicsPathItem, QCheckBox, QFrame, QComboBox, \
    QSlider, QGraphicsEllipseItem
from PyQt5.QtGui import QKeySequence, QPainter, QPen, QPainterPath, \
    QPolygonF, QIntValidator, QDoubleValidator, QColor
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
import pyqtgraph as pg
import imageio as io
from deepd3.core.analysis import Stack, ROI2D_Creator, ROI3D_Creator
from deepd3.core.export import ExportCentroids, ExportImageJ, \
    ExportFolder, ExportPredictions, ExportROIMap
from time import time
from datetime import datetime
import flammkuchen as fl
import json
import pandas as pd
import sys, os
from skimage.color import label2rgb
from scipy.ndimage import gaussian_filter

##############################################
class QHLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

########################
## Ask for dimensions
########################
class askDimensions(QDialog):
    def __init__(self, xy=0.094, z=0.5) -> None:
        """Window asking for dimensions of loaded stack

        Args:
            xy (float, optional): xy dimensions in µm. Defaults to 0.094.
            z (float, optional): z dimensions in µm. Defaults to 0.5.
        """
        super().__init__()

        self.default_xy = xy
        self.default_z = z

        self.l = QGridLayout(self)

        self.l.addWidget(QLabel("Enter your stack dimensions:"))
        self.l.addWidget(QHLine())

        self.xy = QLineEdit()
        self.xy.setPlaceholderText("Default 0.094")
        self.xy.setValidator(QDoubleValidator())
        self.xy.setText(str(self.default_xy))
        self.l.addWidget(QLabel("XY px width in micrometer"))
        self.l.addWidget(self.xy)

        self.z = QLineEdit()
        self.z.setPlaceholderText("Default 0.5")
        self.z.setValidator(QDoubleValidator())
        self.z.setText(str(self.default_z))
        self.l.addWidget(QLabel("Z step in micrometer"))
        self.l.addWidget(self.z)

        self.exec_()

    def dimensions(self):
        """Returns dictionary containing xy and z dimensions in µm

        Returns:
            dict: returns xy and z dimensions
        """
        xy = self.default_xy if self.xy.text() == "" else float(self.xy.text())
        z = self.default_z if self.z.text() == "" else float(self.z.text())

        return dict(xy=xy, z=z)

class ROI3D(QDialog):
    def __init__(self, settings=None):
        """Dialog for 3D ROI build settings
        """
        super().__init__()
        self.l = QGridLayout(self)
        self.go = False

        self.areaThreshold = QLineEdit("0.25")
        self.areaThreshold.setValidator(QDoubleValidator(0, 1, 2))

        self.peakThreshold = QLineEdit("0.80")
        self.peakThreshold.setValidator(QDoubleValidator(0, 1, 2))

        self.minPlanes = QLineEdit("1")
        self.minPlanes.setValidator(QIntValidator(1, 100))
        self.minPx = QLineEdit("20")
        self.minPx.setValidator(QIntValidator(1, 10000))
        self.maxPx = QLineEdit("1000")
        self.maxPx.setValidator(QIntValidator(1, 10000))

        self.distanceToSeed = QLineEdit("2.00")
        self.distanceToSeed.setValidator(QDoubleValidator(0, 20., 2))
        self.seedDelta = QLineEdit("0.2")
        self.seedDelta.setValidator(QDoubleValidator(0, 1, 2))

        self.method = QComboBox()
        self.method.addItems(["floodfill", "connected components"])

        ######## Design
        self.l.addWidget(QLabel("ROI building method"))
        self.l.addWidget(self.method)

        self.l.addWidget(QLabel("Area threshold (0...1)"))
        self.l.addWidget(self.areaThreshold)

        self.l.addWidget(QLabel("Peak threshold (0...1)"))
        self.l.addWidget(self.peakThreshold)

        self.l.addWidget(QHLine())

        self.l.addWidget(QLabel("Minimum planes an ROI should span:"))
        self.l.addWidget(self.minPlanes)

        self.l.addWidget(QLabel("Minimum 3D ROI size [px]:"))
        self.l.addWidget(self.minPx)

        self.l.addWidget(QLabel("Maximum 3D ROI size [px]:"))
        self.l.addWidget(self.maxPx)

        self.l.addWidget(QLabel("Maximum prediction difference to seed pixel prediction"))
        self.l.addWidget(self.seedDelta)

        self.l.addWidget(QLabel("Maximum 3D euclidean distance to seed pixel [px]"))
        self.l.addWidget(self.distanceToSeed)

        self.watershed = QCheckBox("Apply watershed when building ROIs")
        self.l.addWidget(self.watershed)

        self.l.addWidget(QHLine())

        if type(settings) != type(None):
            self.areaThreshold.setText(str(settings['areaThreshold']))
            self.peakThreshold.setText(str(settings['peakThreshold']))
            self.seedDelta.setText(str(settings['seedDelta']))
            self.distanceToSeed.setText(str(settings['distanceToSeed']))
            self.minPx.setText(str(settings['minPx']))
            self.maxPx.setText(str(settings['maxPx']))
            self.minPlanes.setText(str(settings['minPlanes']))
            self.watershed.setChecked(settings['applyWatershed'])

        self.closeButton = QPushButton("Save and start building ROIs")
        self.closeButton.clicked.connect(self.close)
        self.l.addWidget(self.closeButton)

    def close(self):
        self.settings = {
            'method': self.method.currentText(),
            'areaThreshold': float(self.areaThreshold.text()),
            'peakThreshold': float(self.peakThreshold.text()),
            'minPx': int(self.minPx.text()),
            'maxPx': int(self.maxPx.text()),
            'minPlanes': int(self.minPlanes.text()),
            'distanceToSeed': float(self.distanceToSeed.text()),
            'seedDelta': float(self.seedDelta.text()),
            'watershed': bool(self.watershed.isChecked())
        }
        self.go = True 
        super().close()

class ROI2D(QDialog):
    def __init__(self):
        """Dialog for 2D ROI build settings
        """
        super().__init__()
        self.l = QGridLayout(self)
        self.go = False

        self.threshold = QLineEdit("0.25")
        self.threshold.setValidator(QDoubleValidator(0, 1, 2))

        self.l.addWidget(QLabel("Threshold (0...1)"))
        self.l.addWidget(self.threshold)


        self.l.addWidget(QHLine())

        self.cleanROIs = QCheckBox("Clean ROIs")
        self.cleanROIs.setChecked(True)

        self.maxDistanceToDendrite = QLineEdit("30")
        self.maxDistanceToDendrite.setValidator(QIntValidator(0, 100))

        self.minROIsize = QLineEdit("10")
        self.minROIsize.setValidator(QIntValidator(0, 1000))

        self.cleanDendriteThreshold = QLineEdit("0.7")
        self.cleanDendriteThreshold.setValidator(QDoubleValidator(0, 1, 2))

        self.l.addWidget(self.cleanROIs)

        self.l.addWidget(QLabel("Maximum Distance to dendrite [px]"))
        self.l.addWidget(self.maxDistanceToDendrite)

        self.l.addWidget(QLabel("Minimum ROI size [px]"))
        self.l.addWidget(self.minROIsize)

        self.l.addWidget(QLabel("Dendrite threshold"))
        self.l.addWidget(self.cleanDendriteThreshold)

        self.l.addWidget(QHLine())

        self.applyWatershed = QCheckBox("Apply watershed transform")
        self.l.addWidget(self.applyWatershed)

        self.l.addWidget(QHLine())

        self.closeButton = QPushButton("Save and start building ROIs")
        self.closeButton.clicked.connect(self.close)
        self.l.addWidget(self.closeButton)

    def close(self):
        try:
            threshold = float(self.threshold.text())
        except Exception as e:
            QMessageBox.critical(self, "Error with threshold", e)
            return

        try:
            maxD = float(self.maxDistanceToDendrite.text())
        except Exception as e:
            QMessageBox.critical(self, "Error with maximal Distance", e)
            return

        try:
            minS = int(self.minROIsize.text())
        except Exception as e:
            QMessageBox.critical(self, "Error with minimal ROI size", e)
            return

        if not 0 <= threshold <= 1:
            QMessageBox.critical(self, "Wrong threshold", "Threshold should be between 0 and 1!")
            return

        if maxD < 0:
            QMessageBox.critical(self, "Wrong distance", "max Dendrite Distance should be above 0!")
            return    

        self.settings = {
            'threshold': threshold,
            'maxDendriteDistance': maxD,
            'minSize': minS,
            'dendriteThreshold': float(self.cleanDendriteThreshold.text())
        }
        self.go = True 
        super().close()

##############################################
class Segment(QDialog):
    def __init__(self, model_fn=None):
        """Dialog for segmentation settings
        """
        super().__init__()
        self.l = QGridLayout(self)
        self.model_fn = model_fn
        
        self.go = False        

        self.findModelButton = QPushButton("Find...")
        self.findModelButton.clicked.connect(self.findModel)

        self.selectInferenceMode = QComboBox()
        self.selectInferenceMode.addItems([
            'Plane inference',
            'Tile inference [1x]',
            'Tile inference [4x avg]'
        ])

        self.tileSize = QLineEdit("128")
        self.tileSize.setValidator(QIntValidator(32, 128))
        self.insetSize = QLineEdit("96")
        self.insetSize.setValidator(QIntValidator(32, 128))
        # self.fourFoldSegmentation = QCheckBox("Four fold prediction with averaging")
        # self.wholeImageInference = QCheckBox("Whole image inference")


        self.paddingOperation = QComboBox()
        self.paddingOperation.addItems(['min', 'mean']) #("Four fold prediction with averaging")

        self.l.addWidget(QLabel("Select model"))

        if model_fn != None:
            model_fn_ext = model_fn.split("\\")[-1]
            self.l.addWidget(QLabel(f"Current model: \n{model_fn_ext}"))

        self.l.addWidget(self.findModelButton)

        self.l.addWidget(QHLine())

        self.l.addWidget(QLabel("Inference mode"))
        self.l.addWidget(self.selectInferenceMode)

        self.l.addWidget(QHLine())

        self.l.addWidget(QLabel("Tile size (only for tile inference)"))
        self.l.addWidget(self.tileSize)

        self.l.addWidget(QLabel("Inset size (only for tile inference)"))
        self.l.addWidget(self.insetSize)

        self.l.addWidget(QLabel("Padding operation (only for tile inference)"))
        self.l.addWidget(self.paddingOperation)

        self.l.addWidget(QHLine())

        self.runOnCPU = QCheckBox("Run on CPU (uncheck if you want GPU)")
        self.runOnCPU.setChecked(True)
        self.l.addWidget(self.runOnCPU)

        self.closeButton = QPushButton("Save and start segmenting")
        self.closeButton.clicked.connect(self.close)
        self.l.addWidget(self.closeButton)

    def findModel(self):   
        """Find TensorFlow/Keras model on file system
        """
        model_fn = QFileDialog.getOpenFileName(caption="Find TensorFlow Keras model",
            filter="*.h5")

        if model_fn:
            self.model_fn = model_fn

    def close(self):
        if self.model_fn is None:
            QMessageBox.critical(self, 
                "No model selected", 
                "Please select an appropriate model")
            return

        self.settings = {
            'infierenceMode': self.selectInferenceMode.currentText(),
            'tileSize': int(self.tileSize.text()),
            'insetSize': int(self.insetSize.text()),
            'paddingOperation': str(self.paddingOperation.currentText()),
            'runOnCPU': bool(self.runOnCPU.isChecked())
        }

        if self.settings['insetSize'] > self.settings['tileSize']:
            QMessageBox.critical(self, 
                "Settings", 
                "Please choose a insetSize smaller or equal to tileSize")
            return    

        self.go = True 
        super().close()


##############################################
class Cleaning(QDialog):
    previewSignal = pyqtSignal(dict)

    def __init__(self):
        """Dialog for cleaning settings
        """
        super().__init__()
        self.l = QGridLayout(self)
        
        self.go = False        

        ### Closing dendrite
        self.closingDendrite = QCheckBox("Connect single dendrite elements")
        self.closingDendrite.setChecked(True)
        self.closingDendrite.stateChanged.connect(self.previewCleaning)

        self.closingDendriteIterations = QLineEdit("3")
        self.closingDendriteIterations.setValidator(QIntValidator(1, 100))
        self.closingDendriteIterations.textChanged.connect(self.previewCleaning)

        ### Dendrite
        self.cleanDendrite = QCheckBox("Clean dendrite in 3D")
        self.cleanDendrite.setChecked(True)
        self.cleanDendrite.stateChanged.connect(self.previewCleaning)

        self.cleanDendriteThreshold = QLineEdit("0.7")
        self.cleanDendriteThreshold.setValidator(QDoubleValidator(0, 1, 2))
        self.cleanDendriteThreshold.textChanged.connect(self.previewCleaning)

        self.minDendriteSize = QLineEdit("100")
        self.minDendriteSize.setValidator(QIntValidator(1, 10000))
        self.minDendriteSize.textChanged.connect(self.previewCleaning)

        self.l.addWidget(QHLine())

        self.cleanSpines = QCheckBox("Clean spines using dendrite proximity")
        self.cleanSpines.setChecked(True)
        self.cleanSpines.stateChanged.connect(self.previewCleaning)

        self.dendriteDilation = QLineEdit("21")
        self.dendriteDilation.setValidator(QIntValidator(1, 100))
        self.dendriteDilation.textChanged.connect(self.previewCleaning)


        self.preview = QCheckBox("preview")

        self.l.addWidget(QHLine())

        self.l.addWidget(QLabel("Connecting elements"))
        self.l.addWidget(self.closingDendrite)
        self.l.addWidget(QLabel("Connection iterations"))
        self.l.addWidget(self.closingDendriteIterations)

        self.l.addWidget(QHLine())

        self.l.addWidget(QLabel("Clean dendrite segmentation"))
        self.l.addWidget(self.cleanDendrite)
        self.l.addWidget(QLabel("Dendrite threshold"))
        self.l.addWidget(self.cleanDendriteThreshold)
        self.l.addWidget(QLabel("Minimum Dendrite size in px"))
        self.l.addWidget(self.minDendriteSize)

        self.l.addWidget(QHLine())

        self.l.addWidget(QLabel("Clean spine prediction"))
        self.l.addWidget(self.cleanSpines)
        self.l.addWidget(QLabel("Dendrite proximity [dilation iterations]"))
        self.l.addWidget(self.dendriteDilation)

        self.l.addWidget(QHLine())

        self.l.addWidget(self.preview)

        self.l.addWidget(QHLine())

        self.closeButton = QPushButton("Save and start cleaning")
        self.closeButton.clicked.connect(self.close)
        self.l.addWidget(self.closeButton)

    def _settings(self):
        return {
            'closing': self.closingDendrite.isChecked(),
            'closingIterations': int(self.closingDendriteIterations.text() if self.closingDendriteIterations.text() != '' else 1),
            'cleanDendrite': self.cleanDendrite.isChecked(),
            'cleanSpines': self.cleanSpines.isChecked(),
            'dendriteDilation': int(self.dendriteDilation.text() if self.dendriteDilation.text() != '' else 1),
            'cleanDendriteThreshold': float(self.cleanDendriteThreshold.text()),
            'minDendriteSize': int(self.minDendriteSize.text())
        }

    def previewCleaning(self):
        if self.preview.isChecked():
            self.previewSignal.emit(self._settings())

    def close(self):
        self.settings = self._settings()
        
        self.go = True 
        super().close()

###########
## Hack for Double Slider
## https://stackoverflow.com/questions/42820380/use-float-for-qslider
###########
class DoubleSlider(QSlider):
    # create our our signal that we can connect to if necessary
    doubleValueChanged = pyqtSignal(float)

    def __init__(self, decimals=2, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))

##############################################
class testROI(QWidget):
    settings = pyqtSignal(dict)

    def __init__(self, stack, d, s, settings=None) -> None:
        """Tests ROI building in 3D

        Args:
            stack (numpy.ndarray): Part of stack for testing
            d (numpy.ndarray): Dendrite prediction, same shape as `stack`
            s (numpy.ndarray): Spine prediction, same shape as `stack`
            settings (dict, optional): Settings for ROI testing. Defaults to None.
        """
        super().__init__()

        self.stack = stack
        self.d = d 
        self.s = s

        self.l = QGridLayout(self)

        self.l.addWidget(QLabel("Gaussian filter"))
        self.gaussianFilter = DoubleSlider(orientation=Qt.Horizontal)
        self.gaussianFilter.setMinimum(0)
        self.gaussianFilter.setMaximum(3.0)
        self.gaussianFilter.setSingleStep(0.1)
        self.gaussianFilter.setValue(0.0)
        self.gaussianFilter.valueChanged.connect(self.do)
        self.l.addWidget(self.gaussianFilter)

        self.data = QComboBox()
        self.data.addItems(["spine prediction", "intensity"])
        self.data.currentTextChanged.connect(self.do)
        self.l.addWidget(self.data)

        self.mode = QComboBox()
        self.mode.addItems(["floodfill", "connected components"])
        self.mode.currentTextChanged.connect(self.do)
        self.l.addWidget(self.mode)


        self.l.addWidget(QHLine())

        self.l.addWidget(QLabel("Area threshold"))
        self.areaThreshold = DoubleSlider(orientation=Qt.Horizontal)
        self.areaThreshold.setMinimum(0)
        self.areaThreshold.setMaximum(1.0)
        self.areaThreshold.setSingleStep(0.05)
        self.areaThreshold.setValue(0.2)
        self.areaThreshold.valueChanged.connect(self.do)
        self.l.addWidget(self.areaThreshold)

        self.l.addWidget(QLabel("Peak threshold"))
        self.peakThreshold = DoubleSlider(orientation=Qt.Horizontal)
        self.peakThreshold.setMinimum(0)
        self.peakThreshold.setMaximum(1.0)
        self.peakThreshold.setSingleStep(0.05)
        self.peakThreshold.setValue(0.8)
        self.peakThreshold.valueChanged.connect(self.do)
        self.l.addWidget(self.peakThreshold)

        self.l.addWidget(QLabel("Difference to seed intensity [%]"))
        self.seedDelta = DoubleSlider(orientation=Qt.Horizontal)
        self.seedDelta.setMinimum(0)
        self.seedDelta.setMaximum(1.0)
        self.seedDelta.setSingleStep(0.05)
        self.seedDelta.setValue(0.5)
        self.seedDelta.valueChanged.connect(self.do)
        self.l.addWidget(self.seedDelta)

        self.l.addWidget(QLabel("Distance to seed pixel in microns"))
        self.distanceToSeed = DoubleSlider(orientation=Qt.Horizontal)
        self.distanceToSeed.setMinimum(0)
        self.distanceToSeed.setMaximum(10)
        self.distanceToSeed.setSingleStep(0.05)
        self.distanceToSeed.setValue(2)
        self.distanceToSeed.valueChanged.connect(self.do)
        self.l.addWidget(self.distanceToSeed)

        self.l.addWidget(QHLine())

        self.l.addWidget(QLabel("Minimum pixel in ROI"))
        self.minPx = QSlider(orientation=Qt.Horizontal)
        self.minPx.setMinimum(1)
        self.minPx.setMaximum(500)
        self.minPx.setSingleStep(5)
        self.minPx.setValue(10)
        self.minPx.valueChanged.connect(self.do)
        self.l.addWidget(self.minPx)

        self.l.addWidget(QLabel("Maximum pixel in ROI"))
        self.maxPx = QSlider(orientation=Qt.Horizontal)
        self.maxPx.setMinimum(1)
        self.maxPx.setMaximum(10000)
        self.maxPx.setSingleStep(5)
        self.maxPx.setValue(1000)
        self.maxPx.valueChanged.connect(self.do)
        self.l.addWidget(self.maxPx)

        self.l.addWidget(QLabel("Minimum planes in ROI"))
        self.minPlanes = QSlider(orientation=Qt.Horizontal)
        self.minPlanes.setMinimum(1)
        self.minPlanes.setMaximum(10)
        self.minPlanes.setSingleStep(1)
        self.minPlanes.setValue(3)
        self.minPlanes.valueChanged.connect(self.do)
        self.l.addWidget(self.minPlanes)

        self.computeContours = QCheckBox("Compute contours")
        self.computeContours.clicked.connect(self.do)
        self.l.addWidget(self.computeContours)

        self.applyWatershed = QCheckBox("Apply Watershed")
        self.applyWatershed.clicked.connect(self.do)
        self.l.addWidget(self.applyWatershed)

        self.saveBtn = QPushButton("Save settings")
        self.saveBtn.clicked.connect(self.saveSettings)
        self.l.addWidget(self.saveBtn)

        self.imv = pg.ImageView()
        self.imv.setMinimumWidth(600)
        self.imv.setMinimumHeight(400)
        self.l.addWidget(self.imv, 0, 1, 20, 1)
        self.imv.setImage(self.stack.transpose(0,2,1))

        self.overlayItem = pg.ImageItem(np.zeros(self.stack.shape[1:]), 
            compositionMode=QPainter.CompositionMode_Plus)
        self.imv.getView().addItem(self.overlayItem)

        if type(settings) != type(None):
            self.areaThreshold.setValue(settings['areaThreshold'])
            self.peakThreshold.setValue(settings['peakThreshold'])
            self.seedDelta.setValue(settings['seedDelta'])
            self.distanceToSeed.setValue(settings['distanceToSeed'])
            self.minPx.setValue(settings['minPx'])
            self.maxPx.setValue(settings['maxPx'])
            self.minPlanes.setValue(settings['minPlanes'])
            self.applyWatershed.setChecked(settings['applyWatershed'])

        self.do()

        # Update overlay when z location changes
        self.imv.sigTimeChanged.connect(self.changeOverlay)

    def changeOverlay(self):
        ix = self.imv.currentIndex

        self.overlayItem.setImage(self.rgb[ix].transpose(1,0,2))

    def do(self):
        """Actually generating ROIs
        """
        # Uses either intensity or spine prediction
        if self.data.currentText() == 'intensity':
            s = self.stack 
        else:
            s = self.s

        sg = gaussian_filter(s, self.gaussianFilter.value())
        
        self.roi3d = ROI3D_Creator(self.d, sg, 
            mode=self.mode.currentText(),
            areaThreshold=self.areaThreshold.value(),
            peakThreshold=self.peakThreshold.value(),
            seedDelta=self.seedDelta.value(),
            distanceToSeed=self.distanceToSeed.value())

        self.roi3d.computeContours = self.computeContours.isChecked()

        self.roi3d.create(minPx=self.minPx.value(),
            maxPx=self.maxPx.value(),
            minPlanes=self.minPlanes.value(),
            applyWatershed=self.applyWatershed.isChecked())

        self.rgb = label2rgb(self.roi3d.roi_map)

        self.changeOverlay() 

    def saveSettings(self):
        self.settings.emit({
            'areaThreshold': self.areaThreshold.value(),
            'peakThreshold': self.peakThreshold.value(),
            'seedDelta': self.seedDelta.value(),
            'distanceToSeed': self.distanceToSeed.value(),
            'minPx': self.minPx.value(),
            'maxPx': self.maxPx.value(),
            'minPlanes': self.minPlanes.value(),
            'applyWatershed': self.applyWatershed.isChecked()
        })

##############################################
class ImageView(pg.ImageView):
    xy = pyqtSignal(QPointF)
    testROIbuilding = pyqtSignal(QPointF)

    def __init__(self, *args, **kwargs):
        """ImageView - interact with mouse press event
        """
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            # Get xy coordinate in relation to scene
            xy = self.getImageItem().mapFromScene(e.pos())
            # Emit signal to tell interface where mouse click position is
            self.xy.emit(xy)

    def mouseDoubleClickEvent(self, e) -> None:
        xy = self.getImageItem().mapFromScene(e.pos())

        self.testROIbuilding.emit(xy)


class Interface(QWidget):
    def __init__(self, fn, pred_fn, rois_fn, logs_fn, 
            dimensions=dict(xy=0.094, z=0.5)):
        """Main GUI interface

        Args:
            fn (str): path to microscopy stack
            pred_fn (str): path to microscopy stack prediction file
            rois_fn (str): path to microscopy stack ROIs file
            logs_fn (str): path to microscopy stack log file
            dimensions (dict, optional): Dimensions of stack in µm. Defaults to dict(xy=0.094, z=0.5).
        """
        super().__init__()
        self.l = QGridLayout(self)
        self.fn = fn
        self.pred_fn = pred_fn
        self.rois_fn = rois_fn
        self.logs_fn = logs_fn

        self.dimensions = dimensions

        self.log("##################")
        self.log("File opened.")

        self.S = Stack(self.fn, self.pred_fn, dimensions)
        self.selectedRow = -1

        self.settings = None

        # Load ROIs if previously generated and saved...
        if os.path.exists(self.rois_fn):
            r = fl.load(self.rois_fn)
            self.rois = r['rois']
            self.roi_map = r['roi_map']
        else:
            self.rois = None
            self.roi_map = None

        # ROIs currently seen
        self.roisOnImage = []

        # Some presets
        self.currentIndex = 0
        self.showROIs = True
        self.showSegmentation = True
        self.showLabels = False
        self.showMaxProjection = False
        
        # Annotations
        self.annotations = None 
        self.annotationItems = []
        self.annotationColor = QColor(0,0,255, 127)

        # Create an ImageView inside the central widget
        self.imv = ImageView()
        self.imv.setImage(self.S.stack.transpose(0,2,1))
        self.imv.xy.connect(self.roiSelection)
        self.imv.testROIbuilding.connect(self.testROIbuilding)

        # Prediction overlay
        self.overlay = np.zeros(self.S.stack.shape[1:])
        self.overlayItem = pg.ImageItem(self.overlay, compositionMode=QPainter.CompositionMode_Plus)
        self.imv.getView().addItem(self.overlayItem)

        # Update overlay when z location changes
        self.imv.sigTimeChanged.connect(self._changeOverlay)

        self.l.addWidget(self.imv)

        ## ROI TABLE
        self.table = QTableWidget()
        self.table.setMinimumWidth(100)
        self.table.setMaximumWidth(250)
        self.table.setColumnCount(3)

        self.table.setHorizontalHeaderLabels(["Z", "X", "Y"])
        self.table.itemSelectionChanged.connect(self.getSelection)

        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Stretch)

        self.l.addWidget(self.table, 0, 1)

        ## PROGRESS BAR
        self.p = QProgressBar()
        self.p.setMinimumWidth(100)
        self.p.setMaximumWidth(250)
        self.p.setMinimum(0)
        self.p.setMaximum(1)

        self.l.addWidget(self.p, 1, 1)

        ## INFO LABEL BOTTOM LEFT
        self.info = QLabel()
        self.l.addWidget(self.info, 1, 0)

        self.t = []

    def log(self, s):
        with open(self.logs_fn, "a+") as fp:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fp.write(f"{now}: {s}\n")

    def testROIbuilding(self, xy):
        """Test ROI building using dedicated interface.
        Interface is opened at particular stack location where user double-clicked

        Args:
            xy (QPoint): XY Location of pointer during click
        """
        x = int(xy.x())
        y = int(xy.y())
        z = int(self.currentIndex)

        # Stack size for testing, hardcoded
        w = 128
        h = 128

        # Open window of a subset of microscopy stack
        # z-stack +- 3 planes, w/2 left and right to click, h/2 top and bottom to click
        self.testROIwindow = testROI(self.S.stack[z-3:z+3, y-h//2:y+h//2, x-w//2:x+w//2],
            self.S.prediction[z-3:z+3, y-h//2:y+h//2, x-w//2:x+w//2, 0],
            self.S.prediction[z-3:z+3, y-h//2:y+h//2, x-w//2:x+w//2, 1],
            self.settings)
        self.testROIwindow.settings.connect(self.saveSettingsROI3D)
        self.testROIwindow.show()

    def saveSettingsROI3D(self, settings):
        """Save test settings to global settings

        Args:
            settings (dict): 3D ROI settings
        """
        QMessageBox.information(self, "Saved!", "Settings were saved.")
        self.settings = settings

    def drawROIs(self):
        if type(self.rois) is type(None):
            return

        # Remove all items from scene beforehand
        for roi in self.roisOnImage:
            self.imv.getView().removeItem(roi)

        self.roisOnImage = []

        ####### SHOW ROI
        if self.showROIs:
            for i, roi in enumerate(self.rois[self.currentIndex]):
                # Create a path containing the contour
                path = QPainterPath()
                path.addPolygon(QPolygonF([QPointF(*i[::-1]) for i in roi['contour']]))

                # Add path to a graphical item
                roiOnImage = QGraphicsPathItem()
                roiOnImage.setPath(path)
                # White if not selected, yellow if selected, 1px width, solid line
                roiOnImage.setPen(QPen(Qt.white if i != self.selectedRow else Qt.yellow, 1, Qt.SolidLine))

                self.roisOnImage.append(roiOnImage)
                self.imv.getView().addItem(roiOnImage)

    def populateTable(self):
        """Populates ROI table
        """
        # No ROIs available? Do nothing.
        if self.rois is None:
            return 

        # Remove all entries
        self.table.setRowCount(0)

        # Populate table
        for roi in self.rois[self.currentIndex]:
            i = self.table.rowCount()
            self.table.setRowCount(i+1)

            if len(roi['centroid']) == 2:
                y, x = roi['centroid']
                z = float(self.currentIndex)
            else:
                z, y, x = roi['centroid']
            
            self.table.setItem(i, 0, QTableWidgetItem(f"{z:.2f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{x:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{y:.2f}"))

        # And then draw all ROIs
        self.drawROIs()

    def roiSelection(self, xy):
        """Highlight selected ROI due to left click

        Args:
            xy (QPoint): Clicking location
        """
        # Get xy mouse click coordinates
        x = xy.x()
        y = xy.y()

        ds = []


        if self.rois is None:
            return 

        # Compute euclidean distance from mouse click position 
        # to ROI centroid
        for i, roi in enumerate(self.rois[self.currentIndex]):
            if len(roi['centroid']) == 2:
                d = (roi['centroid'][1]-y)**2 + (roi['centroid'][0]-x)**2
            else:
                d = (roi['centroid'][1]-y)**2 + (roi['centroid'][2]-x)**2
            
            ds.append(d)

        # Check if there are any ROIs around...
        if not len(ds):
            return

        # Find closest  
        closest = np.argmin(ds)

        # Highlights ROI closest to click
        self.info.setText(f"z:{self.currentIndex}, x:{x}, y:{y}, closest ROI idx: {closest}")

        # Select element in table and update ROIs
        self.table.selectRow(closest)
        self.drawROIs()

    def getSelection(self):
        """Sets the current row selection in table and updates ROIs,
        because the selected ROI has a different color.
        """
        if len(self.table.selectedIndexes()):
            self.selectedRow = self.table.selectedIndexes()[0].row()
        else:
            self.selectedRow = -1

        self.drawROIs()


    def _changeOverlay(self, z):
        """Hook for z-slider

        Args:
            z (int): current z index
        """
        self.changeOverlay(z)

    def changeOverlay(self, z, preview=False):
        """When the z-slider is changed, update the overlay image (i.e. the prediction)

        Args:
            z (int): current z-location in stack
        """

        # If segmentation should be shown
        if self.showSegmentation and self.showMaxProjection:
            if preview == False:
                self.overlayItem.setImage(self.S.predictionMaxProjection.transpose(1,0,2))
            else:
                self.overlayItem.setImage(self.S.previewMaxProjection.transpose(1,0,2))

        elif self.showSegmentation and not self.showMaxProjection:
            if preview == False:
                self.overlayItem.setImage(self.S.prediction[z].transpose(1,0,2))
            else:
                self.overlayItem.setImage(self.S.preview[z].transpose(1,0,2))


        elif self.showLabels and self.roi_map is not None:
            rm = np.zeros(self.roi_map.shape[1:]+(3,))
            rm[self.roi_map[z] > 0] = (0, 255, 0)
            self.overlayItem.setImage(rm.transpose(1, 0, 2))

        else:
            self.overlayItem.setImage(np.zeros_like(self.S.stack[z]))

        if type(self.annotations) != type(None):
            for i in self.annotationItems:
                self.imv.removeItem(i) 

            self.annotationItems = []

            for i, row in self.annotations.iterrows():
                if 'Z' in row.keys():
                    pos = 'Z'
                else:
                    pos = 'Pos'

                distance = abs(row[pos]-self.currentIndex)

                if distance < 2:
                    size = 2-distance
                    e = QGraphicsEllipseItem(row['X']-size/2, row['Y']-size/2, size, size)
                    e.setBrush(self.annotationColor)
                    e.setPen(QPen(Qt.NoPen))

                    self.annotationItems.append(e)
                    self.imv.addItem(e)

                    
        if self.currentIndex != z:
            # Re-populate table with ROIs from current z-location
            self.populateTable()

        self.drawROIs()

        self.currentIndex = z

    def updateProgress(self, pval, pmax):
        """updates progress bar

        Args:
            pval (int): current value
            pmax (int): target value
        """
        self.t.append(time())

        # Provide ETA
        if len(self.t) > 1:
            dt = abs(np.diff(np.asarray(self.t)).mean())
            self.p.setFormat(f"{int((pval+1)/pmax*100)}% - ETA: {int((pmax-pval+1)*dt)} s")

        # Adjust progress bar settings
        self.p.setMinimum(0)
        self.p.setMaximum(pmax)
        self.p.setValue(pval+1)

        # Check if everything is done
        if pval+1 == pmax:
            self.t = []
            self.p.setFormat("DONE!")

        # Ensure progress bar updates
        QApplication.processEvents()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete:
            q = QMessageBox.question(self, "Delete ROIs", "Do you want to delete the selected ROIs?")

            if q == QMessageBox.Yes:
                checked_rows = []

                # Delete rows from the back to keep order
                for i in self.table.selectedIndexes()[::-1]:
                    cur_row = i.row()

                    if cur_row in checked_rows:
                        continue

                    checked_rows.append(cur_row)

                    # Remove ROI at given location
                    ROI_id = self.rois[self.currentIndex][cur_row]['ROI_id']

                    for z in self.rois.keys():
                        self.rois[z] = [j for j in self.rois[z] if j['ROI_id'] != ROI_id]
                    # self.rois[self.currentIndex].pop(cur_row)

                # Update Table and ROIs in scene
                self.populateTable()
                self.drawROIs()


##############################
##### MAIN WINDOW
##############################
class Main(QMainWindow):
    def __init__(self):
        """Main window for inference GUI
        """
        super().__init__()
        self.status = self.statusBar()
        self.menu = self.menuBar()
        self.model_fn = None

        # Main top menu
        self.file = self.menu.addMenu("&File")
        self.file.addAction("Open", self.open, shortcut=QKeySequence("Ctrl+N"))
        self.file.addAction("Import annotations", self.importAnnotations, shortcut=QKeySequence("Ctrl+I"))
        self.file.addAction("Save", self.save, shortcut=QKeySequence("Ctrl+S"))
        self.file.addAction("Close", self.close)

        self.analyze = self.menu.addMenu("&Analyze")
        self.analyze.setEnabled(False)
        self.analyze.addAction("Segment dendrite and spines", self.segment)
        self.analyze.addAction("Cleaning", self.cleaning)
        self.analyze.addAction("2D ROI detection", self.roi2d)
        self.analyze.addAction("3D ROI detection", self.roi3d)
        self.analyze.addAction("Z projection", self.zprojection)
        self.analyze.addAction("Set dimensions", self.setDimensions)


        self.view = self.menu.addMenu("&View")
        self.view.setEnabled(False)

        self.showROIs = QAction("Show ROIs", self, checkable=True)
        self.showROIs.setChecked(True)
        self.showROIs.triggered.connect(self.setShowROIs)

        self.showSegmentation = QAction("Show Segmentation", self, checkable=True)
        self.showSegmentation.setChecked(True)
        self.showSegmentation.triggered.connect(self.setShowSegmentation)

        self.showMaxProjection = QAction("Show Maximum Projection", self, checkable=True)
        self.showMaxProjection.setChecked(False)
        self.showMaxProjection.triggered.connect(self.setShowMaxProjection)

        self.showLabels = QAction("Show Labels", self, checkable=True)
        self.showLabels.setChecked(False)
        self.showLabels.triggered.connect(self.setShowLabels)

        self.view.addAction(self.showSegmentation)
        self.view.addAction(self.showMaxProjection)
        self.view.addAction(self.showLabels)
        self.view.addAction(self.showROIs)


        self.exportData = self.menu.addMenu("&Export")
        self.exportData.setEnabled(False)
        self.exportData.addAction("Export predictions as tif", self.exportPredictions)
        self.exportData.addAction("Export ROIs to ImageJ", self.exportImageJ)
        self.exportData.addAction("Export ROIs to folder", self.exportToFolderStructure)
        self.exportData.addAction("Export ROI map to file", self.exportRoiMap)
        self.exportData.addAction("Export ROI centroids to file", self.exportRoiCentroids)
        

        # Central widget
        self.w = None

        # Title
        self.setWindowTitle("Interface for spine and dendrite detection")
        self.setGeometry(100, 100, 1200, 600)

    def setShowROIs(self):
        """Toggle ROIs on central widget
        """
        self.w.showROIs = self.showROIs.isChecked()
        self.w.drawROIs()

    def setShowMaxProjection(self):
        """Shows maximum projection of stack and prediction in central widget
        """
        self.w.showMaxProjection = self.showMaxProjection.isChecked()

        if self.w.showMaxProjection:
            self.w.S.predictionMaxProjection = self.w.S.prediction.max(0)
            self.w.S.previewMaxProjection = self.w.S.preview.max(0)

        self.w.changeOverlay(self.w.currentIndex)

    def setShowSegmentation(self):
        """Toggle the segmentation visualization on central widget
        """
        self.w.showSegmentation = self.showSegmentation.isChecked()
        self.w.changeOverlay(self.w.currentIndex)

    def setShowLabels(self):
        """Toggles the visualization of labels on central widget
        """
        self.w.showLabels = self.showLabels.isChecked()
        self.w.showSegmentation = False
        self.showSegmentation.setChecked(False)
        self.w.changeOverlay(self.w.currentIndex)     

    def open(self):
        """Open a z-stack for inference.

        If a prediction and/or ROIs already exist, do load these as well.
        """
        self.fn = QFileDialog.getOpenFileName()[0]

        if self.fn:
            # Create filepaths for related files
            ext = self.fn.split(".")[-1]
            self.pred_fn = self.fn[:-len(ext)-1]+".prediction"
            self.rois_fn = self.fn[:-len(ext)-1]+".rois"
            self.logs_fn = self.fn[:-len(ext)-1]+".log"
            self.roi3d_settings_fn = self.fn[:-len(ext)-1]+".roi3d_settings"

            dim = askDimensions()

            # Create new instance of main widget
            self.status.showMessage(self.fn)
            self.w = Interface(self.fn, self.pred_fn, self.rois_fn, self.logs_fn, dim.dimensions())
            self.setCentralWidget(self.w)

            # Allow the analysis options
            self.analyze.setEnabled(True)
            self.view.setEnabled(True)
            self.exportData.setEnabled(True)

    def importAnnotations(self):
        """Import annotations to visualize those on the central widget
        """
        fn = QFileDialog.getOpenFileName(filter="*.csv")[0]

        if fn:
            df = pd.read_csv(fn, index_col=0)

            self.w.annotations = df 
            self.w.changeOverlay(self.w.currentIndex)

    def save(self):
        """Save segmentation predictions and ROIs
        """
        if type(self.w) == type(None):
            QMessageBox.critical(self, "No file open", "Please open a file first.")
            return

        text = []

        # If segmentation is available
        if self.w.S.segmented:
            fl.save(self.pred_fn, dict(dendrites=self.w.S.prediction[...,0].astype(np.float32),
                spines=self.w.S.prediction[..., 1].astype(np.float32)),
                compression='blosc')
            text.append("Predictions saved!")

        # If ROIs are available
        if self.w.rois:
            fl.save(self.rois_fn, dict(rois=self.w.rois, roi_map=self.w.roi_map),
                compression='blosc')

            text.append("ROIs saved!")

        QMessageBox.information(self, "Saved.", "\n".join(text))


    def segment(self):
        """Segment stack using user-defined settings
        """
        s = Segment(self.model_fn)
        s.exec_()

        if s.go:
            # Only load tensorflow if needed.
            # This is not neccessarily pythonic, but allows for fast GUI loading times
            import tensorflow as tf

            self.w.log("Segmenting data...")

            for k, v in s.settings.items():
                self.w.log(f"{k}: {v}")

            self.w.S.tileSignal.connect(self.w.updateProgress)
            self.model_fn = s.model_fn[0]

            # Decide which hardware will be utilized for inference
            if s.settings['runOnCPU']:
                context = '/cpu:0'

            else:
                context = '/gpu:0'

            with tf.device(context):
                # Predict tiles [4x average]
                if s.selectInferenceMode.currentIndex() == 2:
                    pad_op = np.mean if s.settings['paddingOperation'] == 'mean' else np.min

                    self.w.S.predictFourFold(s.model_fn[0], 
                        tile_size=s.settings['tileSize'],
                        inset_size=s.settings['insetSize'],
                        pad_op=pad_op)

                # Predict the each plane completely in one go
                elif s.selectInferenceMode.currentIndex() == 0:
                    self.w.S.predictWholeImage(s.model_fn[0])
                
                # Predict tiles [1x]
                else:
                    self.w.S.predictInset(s.model_fn[0], 
                        s.settings['tileSize'],
                        s.settings['insetSize'])
    
            self.w.changeOverlay(self.w.currentIndex)

    def cleaning(self):
        """Clean the prediction using user-specified settings
        """
        self.c = Cleaning()
        self.c.previewSignal.connect(self.previewCleaning)
        self.c.exec_()

        if self.c.go:
            print("Actually cleaning...")
            settings = self.c._settings()
            # settings = self.c.settings

            self.w.log("Cleaning stack...")

            for k, v in settings.items():
                self.w.log(f"{k}: {v}")

            if settings['closing']:
                d = self.w.S.closing(settings['closingIterations'])
                self.w.S.prediction[..., 0] = d
                self.w.S.prediction[..., 2] = d

            if settings['cleanDendrite']:
                d = self.w.S.cleanDendrite3D(settings['cleanDendriteThreshold'], settings['minDendriteSize'])
                self.w.S.prediction[..., 0] = d
                self.w.S.prediction[..., 2] = d

            if settings['cleanSpines']:
                s = self.w.S.cleanSpines(settings['cleanDendriteThreshold'], settings['dendriteDilation'])
                self.w.S.prediction[..., 1] = s

            if self.w.showMaxProjection:
                self.w.S.predictionMaxProjection = self.w.S.prediction.max(0)

    def previewCleaning(self, settings):
        """Preview cleaning settings to specify the settings

        Args:
            settings (dict): cleaning settings
        """
        self.w.S.preview = self.w.S.prediction.copy()

        # Clean dendrite in 3D
        if settings['closing']:
            d = self.w.S.closing(settings['closingIterations'], True)
            self.w.S.preview[..., 0] = d
            self.w.S.preview[..., 2] = d

        if settings['cleanDendrite']:
            d = self.w.S.cleanDendrite3D(settings['cleanDendriteThreshold'], settings['minDendriteSize'], True)
            self.w.S.preview[..., 0] = d
            self.w.S.preview[..., 2] = d

        if settings['cleanSpines']:
            s = self.w.S.cleanSpines(settings['cleanDendriteThreshold'], settings['dendriteDilation'], True)
            self.w.S.preview[..., 1] = s

        if self.w.showMaxProjection:
            self.w.S.previewMaxProjection = self.w.S.preview.max(0)

        self.w.changeOverlay(self.w.currentIndex, True)

    def roi2d(self):
        """Create ROIs from segmentation
        """
        if not self.w.S.segmented:
            QMessageBox.critical(self, "No segmentation", "Please provide first a segmentation!")
            return 

        roi = ROI2D()
        roi.exec_()

        if roi.go:
            self.w.log("Building 2D ROIs...")

            for k, v in roi.settings.items():
                self.w.log(f"{k}: {v}")

            r = ROI2D_Creator(self.w.S.prediction[..., 0], # dendrite pred
                              self.w.S.prediction[..., 1], # spines pred
                              roi.settings['threshold']) # Threshold for labelling map
            r.zSignal.connect(self.w.updateProgress)

            # Create ROIs for each z-plane
            self.w.info.setText("Building ROIs...")
            r.create(roi.applyWatershed.isChecked())

            # Clean ROIs using maximum distance to dendrite
            if roi.cleanROIs.isChecked():
                self.w.info.setText("Cleaning ROIs...")
                old, new = r.clean(roi.settings['maxDendriteDistance'], 
                    roi.settings['minSize'], 
                    roi.settings['dendriteThreshold'])

                QMessageBox.information(self, "ROIs cleaned",
                    f"I cleaned all ROIs for you! \nOld rois: {old}\nNew rois: {new}.")

            self.w.roi_map = r.roi_map
            self.w.rois = r.rois
            self.w.populateTable()

    def roi3d(self):
        """Create ROIs from segmentation in 3D
        """
        if not self.w.S.segmented:
            QMessageBox.critical(self, "No segmentation", "Please provide first a segmentation!")
            return 

        roi = ROI3D(self.w.settings)
        roi.exec_()

        if roi.go:
            self.w.log("Building 3D ROIs...")

            for k, v in roi.settings.items():
                self.w.log(f"{k}: {v}")

            with open(self.roi3d_settings_fn, "w+") as fp:
                json.dump(roi.settings, fp)

            r = ROI3D_Creator(self.w.S.prediction[..., 0], # dendrite pred
                              self.w.S.prediction[..., 1], # spines pred
                              roi.settings['method'],
                              roi.settings['areaThreshold'],
                              roi.settings['peakThreshold'],
                              roi.settings['seedDelta'],
                              roi.settings['distanceToSeed'],
                              dimensions=self.w.dimensions) # Threshold for labelling map

            r.zSignal.connect(self.w.updateProgress)
            r.log.connect(self.w.log)

            # Create ROIs for each z-plane
            self.w.info.setText("Building ROIs...")
            r.create(roi.settings['minPx'], 
                    roi.settings['maxPx'],
                    roi.settings['minPlanes'],
                    roi.settings['watershed'])

            # Save results
            self.w.roi_map = r.roi_map
            self.w.rois = r.rois
            self.w.roi_centroids = r.roi_centroids
            self.w.populateTable()

    def setDimensions(self):
        """Set dimensions for z-stack to ensure proper functionality (e.g. distance measures)
        """
        dim = askDimensions(self.w.dimensions['xy'], self.w.dimensions['z'])
        self.w.dimensions = dim.dimensions()
        self.w.S.dimensions = dim.dimensions()

    def zprojection(self):
        """Show a maximum and summed intensity z-projection for the full stack in separate windows
        """
        self.i = pg.image(self.w.S.prediction.max(0).transpose(1,0,2), 
            title="Maximum intensity projection")

        self.j = pg.image(self.w.S.prediction.sum(0).transpose(1,0,2), 
            title="Summed intensity over z")

    def exportImageJ(self):
        """Export ROIs to ImageJ
        """
        fn = QFileDialog.getSaveFileName(caption="Select an ROI file for saving", 
            filter="*.zip")

        if fn[0]:
            self.w.info.setText("Exporting ROIs as ImageJ ROI zip file...")
            eij = ExportImageJ(self.w.rois)
            eij.zSignal.connect(self.w.updateProgress)
            eij.export(fn[0])

            QMessageBox.information(self, "Done!", f"ROIs exported to\n{fn[0]}")

    def exportRoiCentroids(self):
        """Export ROI centroids as CSV file
        """
        fn = QFileDialog.getSaveFileName(caption="Select a CSV file to save the ROI centroids", 
            filter="*.csv")

        if fn[0]:
            e = ExportCentroids(self.w.roi_centroids)
            e.export(fn[0])

            QMessageBox.information(self, "Done!", f"ROI centroids exported to\n{fn[0]}")
            

    def exportRoiMap(self):
        """Export ROI map as TIFF stack
        """
        fn = QFileDialog.getSaveFileName(caption="Select a tif file to save the ROI map", 
            filter="*.tif")

        if fn[0]:
            result = QMessageBox.question(self, "Binarize?", "Should I binarize the ROIs?")

            self.w.info.setText("Exporting ROIs as ROI map to tif file...")
            emap = ExportROIMap(self.w.roi_map, result == QMessageBox.Yes)
            emap.export(fn[0])

            QMessageBox.information(self, "Done!", f"ROI map exported to\n{fn[0]}")
        

    def exportToFolderStructure(self):
        """Export ROIs as folder structure
        """
        folder = QFileDialog.getExistingDirectory(caption="Select folder to export ROIs")

        if not folder:
            return

        ef = ExportFolder(self.w.rois)
        ef.zSignal.connect(self.w.updateProgress)
        ef.export(self.rois_fn, folder)

    def exportPredictions(self):
        """Export neural network prediction as TIFF stacks
        """
        folder = QFileDialog.getExistingDirectory(caption="Select folder to export predictions")

        if not folder:
            return

        ep = ExportPredictions(self.w.S.prediction[..., 1], # spines pred
                              self.w.S.prediction[..., 0]) # dendrite pred
        r, e = ep.export(self.fn, folder)

        if r:
            QMessageBox.information(self, "Exported!", f"Successfully exported to \n{e}")

        else:
            QMessageBox.critical(self, "Failed to export!", f"I had problems to export the data... \n{e}")

def main():
    app = QApplication([])

    m = Main()
    m.show()

    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()