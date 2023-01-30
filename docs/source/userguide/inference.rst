DeepD3 inference
================

Open the inference mode using ``deepd3-inference``. 
Load your stack of choice (we currently support TIF stacks) 
and specify the XY and Z dimensions. Next, you can segment dendrites and 
dendritic spines using a DeepD3 model from `the Model Zoo <https://deepd3.forschung.fau.de/#modelzoo>`_ by clicking 
on ``Analyze -> Segment dendrite and spines``. Afterwards, you may clean 
the predictions by clicking on ``Analyze -> Cleaning``. Finally, you may 
build 2D or 3D ROIs using the respective functions in ``Analyze``. 
To test the 3D ROI building, double click in the stack to a region of interest. 
A window opens that allows you to play with the hyperparameters and segments 
3D ROIs in real-time.

All results can be exported to various file formats. 
For convenience, DeepD3 saves related data in its "proprietary" 
hdf5 file (that you can open using any hdf5 viewer/program/library). 
In particular, you may export the predictions as TIF files, the ROIs to 
ImageJ file format or a folder, the ROI map to a TIF file, or the 
ROI centroids to a file. 

Most functions can be assessed using a batch command script 
located in ``deepd3/inference/batch.py``.