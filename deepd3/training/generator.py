from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout, \
    QSizePolicy, QWidget, QPushButton, QFileDialog, QLineEdit, QDialog, \
    QProgressBar, QMessageBox, QCheckBox, QListWidget, QTreeWidgetItem, QTreeWidget
from PyQt5.QtGui import QPainter, QKeyEvent, QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
import imageio as io
import numpy as np
import os
import flammkuchen as fl
import pandas as pd
from datetime import datetime
from deepd3.core.dendrite import DendriteSWC

class Viewer(QWidget):
    def __init__(self, fn) -> None:
        """View d3data set

        Args:
            fn (str): path to d3data file
        """
        super().__init__()

        self.fn = fn
        self.d = fl.load(fn, "/data")
        self.m = fl.load(fn, "/meta")

        self.l = QGridLayout(self)
        self.imv = pg.ImageView()
        self.imv.setMinimumWidth(800)
        self.imv.setMinimumHeight(800)

        self.imv.setImage(self.d['stack'].transpose(0, 2, 1))
        self.imv.sigTimeChanged.connect(self.plane)

        ### Add overlay
        # Prediction overlay
        self.overlay = np.zeros(self.d['stack'].shape[1:]+(3,), dtype=np.uint8)
        self.overlayItem = pg.ImageItem(self.overlay, compositionMode=QPainter.CompositionMode_Plus)
        self.imv.getView().addItem(self.overlayItem)

        self.plane()

        # TREE
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Meta data'])

        for k, v in self.m.items():
            ki = QTreeWidgetItem([k])
            child = QTreeWidgetItem([str(v)])
            child.setFlags(child.flags() | Qt.ItemIsEditable)
            ki.addChild(child)

            self.tree.addTopLevelItem(ki)

        self.tree.expandToDepth(2)

        self.l.addWidget(self.imv, 0, 0)
        self.l.addWidget(self.tree, 0, 1)

        saveButton = QPushButton("Save")
        saveButton.clicked.connect(self.save)
        self.l.addWidget(saveButton, 1, 1)

    def save(self):
        """Save d3data set
        """
        ok  = QMessageBox.question(self, "Overwrite?",
            "Are you sure? Do you want to overwrite this dataset? There are no sanity checks!")

        if ok != QMessageBox.Yes:
            return

        new_m = dict()

        # Go through each item
        for i in range(self.tree.invisibleRootItem().childCount()):
            c = self.tree.invisibleRootItem().child(i)

            k = c.text(0) 
            v = c.child(0).text(0)

            t = type(self.m[k]) 

            if t == np.float32 or t == np.float64:
                new_m[k] = float(v)

            elif t == np.bool_:
                new_m[k] = True if v.lower() == "true" else False

            elif t == np.int64 or t == np.int32: 
                new_m[k] = int(v)

            else:
                new_m[k] = str(v) 

        new_m["Changed"] = datetime.now().strftime(r"%Y%m%d_%H%M%S")

        fl.save(self.fn, dict(data=self.d, meta=new_m), compression='blosc')

        QMessageBox.information(self, "Saved!",
            f"You saved the data successfully to \n {self.fn}")

    def plane(self):
        """Overlay current annotation plane
        """
        idx = self.imv.currentIndex

        self.overlay[:, :, 0] = self.d['dendrite'][idx].astype(np.uint8) * 255
        self.overlay[:, :, 2] = self.d['dendrite'][idx].astype(np.uint8) * 255
        self.overlay[:, :, 1] = self.d['spines'][idx].astype(np.uint8) * 255

        self.overlayItem.setImage(self.overlay.transpose(1, 0, 2))

class Arrange(QWidget):
    def __init__(self) -> None:
        """Arrange d3data files to a common d3set.
        """
        super().__init__()

        self.setGeometry(100, 100, 400, 600)

        addData = QPushButton("Add data to set")
        addData.clicked.connect(self.addData)

        self.l = QGridLayout(self)
        self.l.addWidget(addData)

        self.l.addWidget(QLabel("Added stacks"))
        
        self.list = QListWidget()
        self.list.setMinimumHeight(200)
        self.l.addWidget(self.list)

        createSet = QPushButton("Create dataset")
        createSet.clicked.connect(self.createSet)

        self.l.addWidget(createSet)

        self.show()

    def keyPressEvent(self, a0) -> None:
        if a0.key() == Qt.Key_Delete:
            print("Del key pressed")
            self.removeSelection()

        return super().keyPressEvent(a0)

    def addData(self):
        """Add selected d3data files
        """
        self.fns = QFileDialog.getOpenFileNames(filter='*.d3data')[0]

        if self.fns:
            for i in self.fns:
                self.list.addItem(i)

    def removeSelection(self):
        """Remove selected d3data sets 
        """
        listItems = self.list.selectedItems()
        
        if not listItems: 
            return        

        ok = QMessageBox.question(self, "Really?",
            "Do you want to delete the selected items?")

        if ok != QMessageBox.Yes:
            return
        
        for item in listItems:
            self.list.takeItem(self.list.row(item))

    def createSet(self):
        """Create d3set from selected d3data files.
        """
        save_fn = QFileDialog.getSaveFileName(caption="Select dataset filename", filter="*.d3set")[0]

        # If the saving fn is ok
        if save_fn:
            stacks = {}
            dendrites = {}
            spines = {}
            meta = pd.DataFrame()

            # For each dataset, add to set
            for i in range(self.list.count()):
                fn = self.list.item(i).text()
                print(fn, "...")

                d = fl.load(fn)

                stacks[f"x{i}"] = d['data']['stack']
                dendrites[f"x{i}"] = d['data']['dendrite']
                spines[f"x{i}"] = d['data']['spines']

                m = pd.DataFrame([d['meta']])
                meta = pd.concat((meta, m), axis=0, ignore_index=True)     

            fl.save(save_fn, dict(data=dict(stacks=stacks, dendrites=dendrites, spines=spines),
                meta=meta), compression='blosc')

            QMessageBox.information(self, "Saved!",
                f"New dataset saved as \n{save_fn}")    


#######################################
class ImageView(pg.ImageView):
    # Signal for first/last plane [0, 1] and current index
    cur_i = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        """Custom Image View for emitting current index w.r.t. key press event
    """
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        # Emit first plane at current index
        if ev == Qt.Key_S:
            self.cur_i.emit(0, self.currentIndex)

        # Emit last plane at current index
        elif ev == Qt.Key_E:
            self.cur_i.emit(1, self.currentIndex)

        return super().keyPressEvent(ev)


########################
## ASK SPACING for DENDRITE TRACINGS
########################
class askSpacing(QDialog):
    def __init__(self) -> None:
        """User interface for spacing w.r.t. dendrite tracing
        """
        super().__init__()

        self.l = QGridLayout(self)

        self.x = QLineEdit()
        self.x.setPlaceholderText("Default: 1")
        self.x.setValidator(QDoubleValidator())
        self.l.addWidget(QLabel("X spacing"))
        self.l.addWidget(self.x)

        self.y = QLineEdit()
        self.y.setPlaceholderText("Default: 1")
        self.y.setValidator(QDoubleValidator())
        self.l.addWidget(QLabel("Y spacing"))
        self.l.addWidget(self.y)

        self.z = QLineEdit()
        self.z.setPlaceholderText("Default: 1")
        self.z.setValidator(QDoubleValidator())
        self.l.addWidget(QLabel("Z spacing"))
        self.l.addWidget(self.z)

        self.exec_()

    def spacing(self):
        """ Converts spacing

        Returns:
            tuple(float, float, float): spacing in µm in x, y and z
        """
        x = 1. if self.x.text() == "" else float(self.x.text())
        y = 1. if self.y.text() == "" else float(self.y.text())
        z = 1. if self.z.text() == "" else float(self.z.text())

        return [x,y,z]


##########################
## Window for creating a dataset for training
## from annotated data
##########################
class addStackWidget(QWidget):
    def __init__(self) -> None:
        """Creates d3data dataset for DeepD3 training
        """
        super().__init__()
        l = QGridLayout(self)
        self.setGeometry(100, 100, 1100, 650)

        self.imv = ImageView()
        self.imv.setMinimumWidth(800)
        self.imv.setMaximumWidth(1400)
        self.imv.cur_i.connect(self.updateZ)

        self.roi = pg.RectROI((100, 100), (200, 300), pen="r")
        self.roi.sigRegionChangeFinished.connect(self.updateROI)
        self.imv.addItem(self.roi)


        ###
        self.stack = None
        self.dendrite     = None 
        self.spines     = None

        ##################
        # Stack
        ##################
        l.addWidget(QLabel("Stack"), 0, 0, 1, 2)
        self.fn_stack = QLabel()
        l.addWidget(self.fn_stack, 1, 0, 1, 2)

        l.addWidget(self.imv, 0, 2, 25, 1)
        
        selectStackBtn = QPushButton("Select stack")
        selectStackBtn.clicked.connect(self.selectStack)
        l.addWidget(selectStackBtn, 2, 0, 1, 2)

        ##################
        # Dendrite tracing
        ##################
        l.addWidget(QLabel("Dendrite tracings"), 3, 0, 1, 2)
        self.fn_d = QLabel("")
        l.addWidget(self.fn_d, 4, 0, 1, 2)
        
        self.selectDendriteBtn = QPushButton("Select dendrite tracings")
        self.selectDendriteBtn.clicked.connect(self.selectDendrite)
        self.selectDendriteBtn.setEnabled(False)
        l.addWidget(self.selectDendriteBtn, 5, 0, 1, 2)

        ##################
        # Spines
        ##################
        l.addWidget(QLabel("Spines"), 6, 0, 1, 2)
        self.fn_s = QLabel("")
        l.addWidget(self.fn_s, 7, 0, 1, 2)
        
        self.selectSpinesBtn = QPushButton("Select spine annotations")
        self.selectSpinesBtn.clicked.connect(self.selectSpines)
        self.selectSpinesBtn.setEnabled(False)
        l.addWidget(self.selectSpinesBtn, 8, 0, 1, 2)

        l.addWidget(QLabel("Resolution"), 9, 0, 1, 2)

        self.res_xy = QLineEdit()
        self.res_xy.setValidator(QDoubleValidator())
        self.res_xy.setPlaceholderText("XY, in microns, e.g. 0.09 for 90 nm resolution in x and y")

        self.res_z = QLineEdit()
        self.res_z.setValidator(QDoubleValidator())
        self.res_z.setPlaceholderText("Z, in microns, e.g. 0.5 for 500 nm step size")

        l.addWidget(self.res_xy, 10, 0, 1, 2)
        l.addWidget(self.res_z, 11, 0, 1, 2)

        l.addWidget(QLabel("Determine offsets using the ROI"), 13, 0, 1, 2)

        self.cropToROI = QCheckBox("Crop annotation to ROI")
        self.cropToROI.setChecked(True)
        l.addWidget(self.cropToROI, 14, 0)

        l.addWidget(QLabel("x"), 15, 0)

        self.offsets_x = QLineEdit("")
        self.offsets_x.setValidator(QIntValidator())
        l.addWidget(self.offsets_x, 15, 1)

        l.addWidget(QLabel("y"), 16, 0)

        self.offsets_y = QLineEdit("")
        self.offsets_y.setValidator(QIntValidator())
        l.addWidget(self.offsets_y, 16, 1)

        l.addWidget(QLabel("w"), 17, 0)

        self.offsets_w = QLineEdit("")
        self.offsets_w.setValidator(QIntValidator())
        l.addWidget(self.offsets_w, 17, 1)

        l.addWidget(QLabel("h"), 18, 0)

        self.offsets_h = QLineEdit("")
        self.offsets_h.setValidator(QIntValidator())
        l.addWidget(self.offsets_h, 18, 1)

        self.zValidator = QIntValidator()
        self.zValidator.setRange(0, 1)

        l.addWidget(QLabel("z (begin), shortcut B"), 19, 0)
        self.offsets_z_begin = QLineEdit("")
        self.offsets_z_begin.setValidator(self.zValidator)
        l.addWidget(self.offsets_z_begin, 19, 1)

        l.addWidget(QLabel("z (end), shortcut E"), 20, 0)
        self.offsets_z_end = QLineEdit("")
        self.offsets_z_end.setValidator(self.zValidator)
        l.addWidget(self.offsets_z_end, 20, 1)

        self.progressbar = QProgressBar()

        l.addWidget(self.progressbar, 21, 0, 1, 2)

        saveBtn = QPushButton("Save annotation stack")
        saveBtn.clicked.connect(self.save)
        l.addWidget(saveBtn, 22, 0, 1, 2)


        expand = QLabel()
        sizePolicy = QSizePolicy(QSizePolicy.Expanding , QSizePolicy.Expanding )
        expand.setSizePolicy(sizePolicy)

        l.addWidget(expand, 24, 0)

    def updateZ(self, a, b):
        """Updates z-level in stack

        Args:
            a (int): z-stack begin, first plane
            b (int): z-stack end, last plane
        """
        if a == 0:
            self.offsets_z_begin.setText(str(b))

        else:
            self.offsets_z_end.setText(str(b))

    def keyPressEvent(self, a0: QKeyEvent) -> None:
        """Key press event to enable shortcuts

        Args:
            a0 (QKeyEvent): Key event

        """
        if a0.key() ==  Qt.Key_S or a0.key() == Qt.Key_B:
            self.offsets_z_begin.setText(str(self.imv.currentIndex))

        elif a0.key() == Qt.Key_E:
            self.offsets_z_end.setText(str(self.imv.currentIndex))

        return super().keyPressEvent(a0)

    def save(self):
        """Save a d3data set
        """
        fn = QFileDialog.getSaveFileName(filter="*.d3data")[0]

        if not fn:
            return

        x = int(self.offsets_x.text())
        y = int(self.offsets_y.text())
        w = int(self.offsets_w.text())
        h = int(self.offsets_h.text())
        z_begin = int(self.offsets_z_begin.text())
        z_end = int(self.offsets_z_end.text())

        if x < 0:
            x = 0

        if y < 0:
            y = 0

        # Check for maximum size
        if x+w >= self.im.shape[2]:
            w = self.im.shape[2]-x-1

        if y+h >= self.im.shape[1]:
            h = self.im.shape[1]-y-1

        if z_end - z_begin < 0 or z_begin < 0 or z_end >= self.im.shape[0]:
            QMessageBox.critical(self, "Z span invalid",
            "Please check for z_begin and z_end.")

        try:
            res_xy = float(self.res_xy.text())
            res_z = float(self.res_z.text())
        except Exception as e:
            QMessageBox.critical(self, "Something went wrong",
                f"{e}")
            return

        if self.cropToROI.isChecked():
            stack = self.im[z_begin:z_end+1, y:y+h, x:x+w]
            dendrite = self.dendrite[z_begin:z_end+1, y:y+h, x:x+w]
            spines = self.spines[z_begin:z_end+1, y:y+h, x:x+w]

        else:
            stack = self.im[z_begin:z_end+1]
            dendrite = self.dendrite[z_begin:z_end+1]
            spines = self.spines[z_begin:z_end+1]


        data = {
            'stack': stack,
            'dendrite': dendrite > 0,
            'spines': spines > 0
        }

        meta = {
            'crop': self.cropToROI.isChecked(),
            'X': x,
            'Y': y,
            'Width': w,
            'Height': h,
            'Depth': z_end-z_begin+1,
            'Z_begin': z_begin,
            'Z_end': z_end,
            'Resolution_XY': res_xy,
            'Resolution_Z': res_z,
            'Timestamp': datetime.now().strftime(r"%Y%m%d_%H%M%S"),
            'Generated_from': self.fn_stack.text()
        }

        try:
            fl.save(fn, dict(data=data, meta=meta), compression='blosc')
            QMessageBox.information(self, "Data saved",
            f"Your data was successfully saved:\n{fn}")

        except Exception as e:
            QMessageBox.critical(self, "Could not save data",
            f"{e}")        
        

    def updateROI(self):
        """Updates the ROI chosen as dataset
        """
        # Retrieve xy location and ROI rectangle size
        pos = int(self.roi.pos().x()), int(self.roi.pos().y())  # x, y
        size = int(self.roi.size().x()), int(self.roi.size().y())

        # Update the offset fields
        self.offsets_x.setText(str(pos[0]))
        self.offsets_y.setText(str(pos[1]))
        self.offsets_w.setText(str(size[0]))
        self.offsets_h.setText(str(size[1]))          


    def selectStack(self):
        """Select a microscopy stack
        """
        fn = QFileDialog.getOpenFileName(caption="Select stack", filter="*.tif")[0]

        if fn:
            print(fn)
            self.progressbar.setMaximum(10)
            self.fn_stack.setText(fn)
            self.im = np.asarray(io.mimread(fn, memtest=False))
            self.progressbar.setValue(9)
            self.imv.setImage(self.im.transpose(0, 2, 1))
            self.progressbar.setValue(10)

            ### Enable other buttons
            self.selectDendriteBtn.setEnabled(True)
            self.selectSpinesBtn.setEnabled(True)

            ### Add overlay
            # Prediction overlay
            self.overlay = np.zeros(self.im.shape[1:]+(3,), dtype=np.uint8)
            self.overlayItem = pg.ImageItem(self.overlay, compositionMode=QPainter.CompositionMode_Plus)
            self.imv.getView().addItem(self.overlayItem)

            # Update overlay when z location changes
            self.imv.sigTimeChanged.connect(self.changeOverlay)

            self.offsets_z_begin.setText("0")
            self.offsets_z_end.setText(str(self.im.shape[0]-1))

            self.zValidator.setRange(0, self.im.shape[0]-1)

    def changeOverlay(self):
        """Show the dendrite and spine annotations as overlay in addition to the original stack
        """
        self.overlay = np.zeros_like(self.overlay)

        # current z index
        cur_i = self.imv.currentIndex 

        # if dendrite segmentation is available
        if type(self.dendrite) != type(None):
            self.overlay[..., 0] = self.dendrite[cur_i]
            self.overlay[..., 2] = self.dendrite[cur_i] 

        # if spines segmentation is available
        if type(self.spines) != type(None):
            self.overlay[..., 1] = self.spines[cur_i] * 255

        # Show the image
        self.overlayItem.setImage(self.overlay.transpose(1, 0, 2))

    ###########################################
    def selectDendrite(self):
        """Select dendrite annotation file
        """
        fn = QFileDialog.getOpenFileName(caption="Select dendrite tracings", filter="*.tif; *.swc")[0]

        if fn:
            self.fn_d.setText(fn)

            if fn.endswith("swc"): 
                target_fn = fn[:-4] + "_dendrite.tif"

                if os.path.exists(target_fn):
                    ok = QMessageBox.question(self, 
                        "Keep it?", 
                        "We found an existing converted dendritic trace. Should I keep it?")

                    if ok == QMessageBox.Yes:
                        self.dendrite = np.asarray(io.mimread(target_fn, memtest=False))
                        return 

                aS = askSpacing()

                ### Now convert dendrite
                d = DendriteSWC(spacing=aS.spacing())
                d.node.connect(self.updateProgress)
                d.open(fn, self.fn_stack.text())
                d.convert(target_fn)

                self.dendrite = np.asarray(io.mimread(target_fn, memtest=False))

            else:
                self.dendrite = np.asarray(io.mimread(fn, memtest=False))

    def updateProgress(self, a, b):
        """Update progress bar

        Args:
            a (int, float): maximum of progress bar
            b (int, float): current value of progress bar
        """
        self.progressbar.setMaximum(b)
        self.progressbar.setValue(a)

    def selectSpines(self):
        """Select a spine annotation
        """
        fn = QFileDialog.getOpenFileName(caption="Select stack", filter="*.tif, *.mask")[0]

        if fn:
            self.fn_s.setText(fn)

            if fn.endswith("mask"):
                # Load a pipra annotated mask file
                mask = fl.load(fn, "/mask").transpose(0, 2, 1)

            else:
                # Load a tif file
                mask = np.asarray(io.mimread(fn, memtest=False))

            self.spines = mask


class Selector(QWidget):
    def __init__(self):
        """Select a given task in the DeepD3 training pipeline
        """
        super().__init__()

        l = QGridLayout(self)

        add = QPushButton("Create training data")
        add.setFixedWidth(300)
        add.setFixedHeight(100)
        add.clicked.connect(self.createTrainingData)

        l.addWidget(add)

        view = QPushButton("View training data")
        view.setFixedWidth(300)
        view.setFixedHeight(100)
        view.clicked.connect(self.viewTrainingData)

        l.addWidget(view)

        arrange = QPushButton("Arrange training data")
        arrange.setFixedWidth(300)
        arrange.setFixedHeight(100)
        arrange.clicked.connect(self.arrangeTrainingData)

        l.addWidget(arrange)

    def viewTrainingData(self):
        """View training data
        """
        fn = QFileDialog.getOpenFileName(filter="*.d3data")[0]

        if fn:
            # If a valid filename was given
            self.c = Viewer(fn)
            self.c.show()

    def createTrainingData(self):
        """Create d3data set
        """
        self.a = addStackWidget()
        self.a.show()

    def arrangeTrainingData(self):
        """Arrange training data (d3data files) in a d3set
        """
        self.b = Arrange()
        self.b.show()

def main():
    """Main entry point to GUI
    """
    app = QApplication([])
    
    # Select which part of the DeepD3 training pipeline is used.
    s = Selector()
    s.show()

    app.exec_()

if __name__ == '__main__':
    main()