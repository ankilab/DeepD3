import sys
import argparse
import tensorflow as tf
import flammkuchen as fl
import numpy as np
from deepd3.core.analysis import Stack, ROI2D_Creator, ROI3D_Creator

parser = argparse.ArgumentParser()
parser.add_argument('stack', type=argparse.FileType('r', encoding='UTF-8'), help='File to be segmented')
parser.add_argument('neuralnetwork', metavar='nn', type=argparse.FileType('r', encoding='UTF-8'), help='Deep neural network for spine and dendrite segmentation')
parser.add_argument('--tile_size', metavar='ts', type=int, help='Tile size for segmentation (default: 128)', default=128)
parser.add_argument('--inset_size', metavar='is', type=int, help='Inset size for segmentation (default: 96)', default=96)
parser.add_argument('--average', action='store_const', const=True, help='Predict segmentation with four offsets, average. (default: False)', default=False)
parser.add_argument('--plane', action='store_const', const=True, help='Predict segmentation in a whole plane at once. (default: False)', default=False)
parser.add_argument('--clean_dendrite', action='store_const', const=True, help='Clean dendrite using connected components per plane (2D). (default: False)', default=False)
parser.add_argument('--clean_dendrite_3d', action='store_const', const=True, help='Clean dendrite using connected components in three dimensions (3D). (default: True)', default=True)
parser.add_argument('--min_dendrite_size',  type=int, help='Minimum dendrite element size in px for cleaning (default: 100)', default=100)
parser.add_argument('--dendrite_threshold',  type=float, help='Dendrite probability threshold for cleaning (default: 0.7)', default=0.7)
parser.add_argument('--dendrite_dilation',  type=float, help='Dendrite dilation factor for spine cleaning (default: 11)', default=11)
parser.add_argument('--clean_spines',  action='store_const', const=True, help='Clean spines using dendrite prediction dilation. (default: True)', default=True)


parser.add_argument('--build_rois_2d', action='store_const', const=True, help='Enable 2D ROI building')
parser.add_argument('--build_rois_3d', action='store_const', const=True, help='Enable 3D ROI building')

parser.add_argument('--roi_method',  type=str, help='ROI building method: floodfill or connected components (default: floodfill)', default="floodfill")
parser.add_argument('--roi_areaThreshold',  type=float, help='ROI probability threshold for area (default: 0.25)', default=0.25)
parser.add_argument('--roi_peakThreshold',  type=float, help='ROI probability threshold for peak (default: 0.80)', default=0.80)
parser.add_argument('--roi_seedDelta',  type=float, help='Pixel similarity to seed pixel (default: 0.2)', default=0.2)
parser.add_argument('--roi_distanceToSeed',  type=float, help='Distance to seed pixel in px (default: 10)', default=10)

parser.add_argument('--watershed',  action='store_const', const=True, help='Apply watershed (default: False)', default=False)
parser.add_argument('--clean_rois', action='store_const', const=True, help='Enable ROI cleaning')
parser.add_argument('--min_roi_size', type=int, help='Minimum ROI size in px (default: 10)', default=10)
parser.add_argument('--max_roi_size', type=int, help='Maximum ROI size in px (default: 1000)', default=1000)
parser.add_argument('--min_planes', type=int, help='Minimum Planes an ROI should span (default: 1)', default=1)

if __name__ == '__main__':
    args = parser.parse_args()

    # Define filenames
    fn = args.stack.name
    ext = fn.split(".")[-1]
    pred_fn = fn[:-len(ext)-1]+".prediction"
    rois_fn = fn[:-len(ext)-1]+".rois"


    print("Loading stack...")
    S = Stack(fn)

    if args.average:
        print("Predicting inset four times, average")
        S.predictFourFold(args.neuralnetwork.name,
                args.tile_size, 
                args.inset_size)

    elif args.plane:
        print("Predict whole image in plane")
        S.predictWholeImage(args.neuralnetwork.name)

    else:
        print("Predicting inset")
        S.predictInset(args.neuralnetwork.name,
                    args.tile_size, 
                    args.inset_size)

    
    if args.clean_dendrite:
        print("Cleaning dendrite in 2D")
        d = S.cleanDendrite(args.dendrite_threshold, args.min_dendrite_size)
        S.prediction[..., 0] = d
        S.prediction[..., 2] = d

    if args.clean_dendrite_3d:
        print("Cleaning dendrite in 3D")
        S.cleanDendrite3D(args.dendrite_threshold, args.min_dendrite_size)
        S.prediction[..., 0] = d
        S.prediction[..., 2] = d

    if args.clean_spines:
        print("Cleaning spines")
        s = S.cleanSpines(args.dendrite_threshold, args.dendrite_Dilation)
        S.prediction[..., 1] = s

    print("Saving predictions")
    fl.save(pred_fn, 
            dict(dendrites=S.prediction[...,0].astype(np.float32),
                spines=S.prediction[..., 1].astype(np.float32)),
                compression='blosc')


    if args.build_rois_2d:
        print("Building 2D ROIs...")

        if args.build_rois_3d:
            print("********************")
            print("Caution: You also selected 3D ROI building, ", end="")
            print("please disable then 2D ROI building.")
            print("No 3D ROIs will be built.")
            print("********************")

        r = ROI2D_Creator(S.prediction[..., 0],
            S.prediction[..., 1],
            args.roi_threshold)

        r.create(args.watershed)

        if args.clean_rois:
            print("Cleaning 2D ROIs...")
            r.clean(args.max_dendrite_displacement,
                    args.min_roi_size,
                    args.dendrite_threshold)

        print("Saving 2D ROIs...")
        fl.save(rois_fn, 
                dict(rois=r.rois, roi_map=r.roi_map),
                compression='blosc')

    elif args.build_rois_3d:
        print("Building 3D ROIs...")
        r = ROI3D_Creator(S.prediction[..., 0],
            S.prediction[..., 1],
            args.method,
            args.roi_areaThreshold,
            args.roi_peakThreshold,
            args.roi_seedDelta,
            args.roi_distanceToSeed)

        r.create(args.min_roi_size,
            args.max_roi_size,
            args.min_planes)

        print("Saving ROIs...")
        fl.save(rois_fn, 
                dict(rois=r.rois, roi_map=r.roi_map),
                compression='blosc')

    print("Done!")

    