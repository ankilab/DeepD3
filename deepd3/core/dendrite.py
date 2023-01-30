import numpy as np
import pandas as pd
import imageio as io
from tqdm import tqdm
import pathlib
import numpy as np
from skimage.draw import disk
from PyQt5.QtCore import QObject, pyqtSignal

            
def line_w_sphere(s, p0, p1, r0, r1, color=1, spacing=[1, 1, 1]):
    """Draw a line with width in 3D space

    Args:
        s (numpy.ndarray): the 3D stack
        p0 (tuple): point 0 (x, y, z)
        p1 (tuple): point 1 (x, y, z)
        r0 (float): radius for point 0
        r1 (float): radius for point 1
        color (int, optional): Color for drawing, e.g. 255 for np.uint8 stack. Defaults to 1.
        spacing (list, optional): Spacing in 3D (x,y,z). Defaults to [1, 1, 1].
    """
    assert len(p0) == len(p1), "points should have same depth"

    if len(s.shape) == 2:
        s = s[None]
        p0 = p0[0], p0[1], 0
        p1 = p1[0], p1[1], 0

    assert len(s.shape) == 3, "stack s should be 2D or 3D"

    # Unpack coordinates
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x, y, z = p0

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    sz = -1 if z0 > z1 else 1

    derr = max([dx, dy, dz])
    
    # Interpolate the differences and the radii for drawing
    r = np.interp(range(derr), [0, derr-1], [r0, r1])

    errx = derr / 2
    erry = derr / 2
    errz = derr / 2

    for i in range(derr):
        # draw point in line
#         s[z, x, y] = color
        # draw sphere with radius r
        sphere(s, [x, y, z], r[i]*2, spacing=spacing, color=color)

        # Update coordinates
        errx -= dx

        if errx < 0:
            x += sx
            errx += derr

        erry -= dy

        if erry < 0:
            y += sy
            erry += derr

        errz -= dz

        if errz < 0:
            z += sz
            errz += derr


def sphere(s, p0, d, spacing=[1, 1, 1], color=255, debug=False):
    """Draw a 3D sphere with given diameter d at point p0 in given color.

    Args:
        s (numpy.ndarray): numpy 3D stack
        p0 (tuple): x, y, z tuple
        d (float): diameter in 1 spacing unit
        spacing (list, optional): x, y, z spacing; x and y spacing must be equal. Defaults to [1, 1, 1].
        color (int, optional): Draw color, e.g. 255 for np.uint8 stack. Defaults to 255.
        debug (bool, optional): if True prints plane related information. Defaults to False.
    """
    assert spacing[0] == spacing[1], "x and y spacing must be the same!"

    # Convert to pixels
    d_xy = d / spacing[0]
    r = d_xy / 2

    # Initialize center
    x, y, z = p0

    # Iterate over planes where the sphere is visible
    for plane in range(z - int(d / spacing[2] / 2), z + int(d / spacing[2] / 2) + 1):
        radius = np.sqrt((d / 2) ** 2 - ((plane - z) * spacing[2]) ** 2) / spacing[0]

        if debug:
            print(plane, (plane - z) * spacing[2], "µm to center")
            print(radius * spacing[0], "µm, ", np.round(radius, 3), "px\n")

        # If sphere is to be drawn
        if radius > 0:
            # Draw a circle on a diameter x diameter grid w/ given radius
            rr, cc = disk((d_xy//2, d_xy//2), radius, shape=(d_xy, d_xy))

            if plane < 0 or plane >= s.shape[0]:
                continue

            if (rr + x - d_xy//2).max() >= s.shape[1] or (cc + y - d_xy//2).max() >= s.shape[2]:
                continue

            # Go to plane in stack and move circle to right position, acts in-place
            # different to previous shift with circle somehow...
            s[plane, (rr + x - d_xy//2).astype(int), (cc + y - d_xy//2).astype(int)] = color

def xyzr(swc, i):
    """returns xyz coordinates and radius as tuple from swc pandas DataFrame and loc i,
    actually it is y, x and z

    Args:
        swc (pandas.DataFrame): List of traced coordinates
        i (int): current location

    Returns:
        tuple: y, x, z and r coordinates as integers
    """
    return (int(swc.loc[i].y), int(swc.loc[i].x), int(swc.loc[i].z)), int(swc.loc[i].r)
            
class DendriteSWC(QObject):
    node = pyqtSignal(int, int)

    def __init__(self, spacing=[1, 1, 1]):
        """Converting a neuron trace saved as swc file back to a clean tif stack

        Args:
            spacing (list, optional): Spacing in 3D (x, y, z). Defaults to [1, 1, 1].
        """
        super().__init__()

        self.swc = None
        self.ref = None
        self.spacing = spacing

    def open(self, swc_fn, ref_fn):
        """Open and read the swc and the stack file.

        Args:
            swc_fn (str): The file path to the swc file
            ref_fn (str): The file path to the stakc file
        """

        print('Check for comments in SWC file...')
        
        skiprows = 0
        
        with open(swc_fn) as fp:
            while True:
                line = fp.readline()
                if line.startswith("#"):
                    skiprows += 1
                    print(line.strip())

                else:
                    break
        print(f'   --> will skip {skiprows} rows.')

        print('Load SWC file...')

        self.swc = pd.read_csv(swc_fn,
                  sep=' ',
                  header=None,
                  skiprows=skiprows,
                  index_col=0,
                  names=('idx','kind','x','y','z','r','parent'))

        print('Load ref file...')
        self.ref = np.asarray(io.mimread(ref_fn, memtest=False)) 

        print(self.ref.shape)

        self.ref_fn = ref_fn
        self.swc_fn = swc_fn


    def convert(self, target_fn=None):
        """Convert swc file to tif stack

        Args:
            target_fn (str, optional): Target path. Defaults to None.

        Returns:
            return: save path
        """
        print('Create stack...')
        self.stack = np.zeros(self.ref.shape, dtype=np.uint8)

        print('Binarize SWC...')
        self._binarize_swc_w_spheres()

        if target_fn is None:
            target = self.ref_fn if target_fn is None else target_fn
            path_ref = pathlib.Path(target)
            ext = path_ref.suffix

            save_fn = target.replace(ext, f"_dendrite.tif")
            save_fn_max = target.replace(ext, f"_dendrite_max.png")

        else:
            path_ref = pathlib.Path(target_fn)
            ext = path_ref.suffix

            save_fn = target_fn 
            save_fn_max = target_fn.replace(ext, f"_max.png")

        print('Saving stack and maximum intensity projection')
        io.mimwrite(save_fn, self.stack)
        io.imwrite(save_fn_max, self.stack.max(0))

        print(f'Data saved as {save_fn} \n and \n {save_fn_max}')
        print()

        return save_fn

    def _binarize_swc_w_spheres(self):
        '''Binarizes SWC file in a given 3D stack with spheres'''
    
        for i in tqdm(range(1, self.swc.shape[0])):
            if self.swc.loc[i].parent > 0:
                p0, r0 = xyzr(self.swc, self.swc.loc[i].parent)
                p1, r1 = xyzr(self.swc, i)
                
                try:
                    line_w_sphere(self.stack, p0, p1, r0, r1, 255, self.spacing)
                except:
                    pass

                self.node.emit(i, self.swc.shape[0])

if __name__ == '__main__':
    pass