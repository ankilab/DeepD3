import numpy as np
from numba import njit
import pandas as pd

@njit
def _computeDistance(p1, p2, dxy=0.1, dz=0.5):
    """compute euclidean distance of two points in space. Points are in (Z, Y, X) format"""
    di_z = (p1[0] * dz - p2[0] * dz) ** 2
    di_y = (p1[1] * dxy - p2[1] * dxy) ** 2
    di_x = (p1[2] * dxy - p2[2] * dxy) ** 2
    
    return np.sqrt(di_z + di_y + di_x)

@njit
def _distanceMatrix(pt1, pt2, dxy=0.1, dz=0.5) -> np.ndarray:
    """Compute distance matrix of points in 3D (Z, Y, X).
    Works only on 3D data

    Args:
        pt1 (numpy.ndarray): Points to be matched
        pt2 (numpy.ndarray): Points that can be matched
        dxy (float, optional): Pitch in xy. Defaults to 0.1.
        dz (float, optional): Pitch in z. Defaults to 0.5.

    Returns:
        numpy.ndarray: distance map from pt1 and pt2 points
    """
    dm = np.zeros((pt1.shape[0], pt2.shape[0]))
    
    for i in range(pt1.shape[0]):
        for j in range(pt2.shape[0]):
            dm[i, j] = _computeDistance(pt1[i], pt2[j], dxy=dxy, dz=dz)
            
    return dm
    
def distanceMatrix(pt1, pt2, dxy=0.1, dz=0.5):
    """Compute distance matrix of points in 2D (Y, X) and 3D (Z, Y, X)

    Args:
        pt1 (numpy.ndarray): Points to be matched
        pt2 (numpy.ndarray): Points that can be matched
        dxy (float, optional): Pitch in xy. Defaults to 0.1.
        dz (float, optional): Pitch in z. Defaults to 0.5.
    """
    if pt1.shape[1] == 2:
        pt1 = np.insert(pt1, 0, 1, axis=1)
        
    if pt2.shape[1] == 2:
        pt2 = np.insert(pt2, 0, 1, axis=1)
        
    return _distanceMatrix(pt1, pt2, dxy=dxy, dz=dz)

def _countOccurences(arr) -> dict:
    """Count occurences in array

    Args:
        arr (numpy.ndarray): Array with non-unique numbers

    Returns:
        dict: Dictionary with unique numbers as keys and their occurence as value
    """
    d = dict()
    
    for i in arr:
        if i not in d.keys():
            d[i] = 1
            
        else:
            d[i] += 1
            
    return d

def findMatches(pt1, pt2, dxy=0.1, dz=0.5, threshold_distance=1.2):
    matched = []
    unmatched = []

    # Compute distance Matrix
    dm = distanceMatrix(pt1, pt2, dxy=dxy, dz=dz)

    # Find minimal distances and point ids
    min_di = np.min(dm, 1)
    min_pt = np.argmin(dm, 1)

    occurences = _countOccurences(min_pt)

    assigned_pts = []

    # Iterate over pt1 points
    for i in range(pt1.shape[0]):
        
        # Point is close in space and was uniquely assigned
        if min_di[i] < threshold_distance and occurences[min_pt[i]] == 1:
            matched.append([
                pt1[i],
                pt2[min_pt[i]]
            ])
            
            assigned_pts.append(min_pt[i])
            
        # Point is close in space and was assigned multiple times        
        elif min_di[i] < threshold_distance and occurences[min_pt[i]] > 1:
            # The current point is the closest to the assigned point
            # and has not been assigned just yet (e.g. two points exactly the same distance)
            if min_di[min_pt == min_pt[i]].min() == min_di[i] and min_pt[i] not in assigned_pts:
                matched.append([
                    pt1[i],
                    pt2[min_pt[i]]
                ])
                
                assigned_pts.append(min_pt[i])
                
            # Remove from list, because...
            else:
                min_pt[i] = -1
                unmatched.append(pt1[i])
                
        # Point could not be matched for some reason
        else:
            unmatched.append(pt1[i])
            
    return matched, unmatched, assigned_pts

def createMatchMap(matched, safety=1.2):
    to_df = []
    
    for p1, p2 in matched:
        if p1.size == 2:
            p1 = np.insert(p1, 0, 0)

        if p2.size == 2:
            p2 = np.insert(p2, 0, 0)

        c = (p1+p2)/2
        
        to_df.append({
            'Z1': p1[0],
            'Z2': p2[0],
            'Y1': p1[1],
            'Y2': p2[1],
            'X1': p1[2],
            'X2': p2[2],
            'CZ': c[0],
            'CY': c[1],
            'CX': c[2],
            'R': np.sqrt(np.sum((p2[1:]-p1[1:])**2)) * safety
        })
    
    return pd.DataFrame(to_df)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import flammkuchen as fl

    ####### 
    # Create some fake data
    im = np.zeros((64, 64, 3), dtype=np.int32)

    pt1 = [
        (5, 5),
        (23, 24),
        (15, 12),
        (3, 40),
        (60, 60),
        (45, 10),
        (35, 10)
    ]

    pt2 = [
        (5, 7),
        (25, 24),
        (18, 15),
        (49, 51),
        (60, 61),
        (61, 60),
        (40, 10)
    ]

    for p in pt1:
        im[p] += (255, 0, 255)
        
    for p in pt2:
        im[p] += (0, 255, 0)
        
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)
        
    im = np.uint8(im)

    #####
    # Compute matching and create match map

    matched, unmatched, assigned_pts = findMatches(pt1, pt2)

    print(matched)

    df = createMatchMap(matched)
    df.to_csv("matched_points_test.matched")
    fl.save("matched_points_test.h5", dict(stack=im[None]))

    ######
    # Show the data
    plt.figure()
    ax = plt.subplot(111)
    plt.imshow(im)

    for i, row in df.iterrows():
        c = plt.Circle((row.CX, row.CY), row.R, color='b', fill=False)
        ax.add_patch(c)

    plt.show()  
        