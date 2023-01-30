import flammkuchen as fl
import numpy as np
import imageio as io
import pathlib

class Spines:
    def __init__(self):
        """Spine annotation data processing
        """
        self.spines_fn = None
        
    def open(self, spines_fn : str):
        """Saves path to object

        Args:
            spines_fn (str): path to spines annotation file
        """
        self.spines_fn = spines_fn 

    def convert(self):
        """Loads and converts spine annotation files to TIFF stacks

        Returns:
            str: Path to saved TIFF stack
        """
        # Mask drawn in e.g. ImageJ and saved as TIF file
        if self.spines_fn.endswith('tif'):
            stack = io.mimread(self.spines_fn)

        # Mask drawn using pipra and saved as mask file
        elif self.spines_fn.endswith('mask'):
            stack = fl.load(self.spines_fn)['mask']
            stack = stack.astype(np.uint8).transpose(0,2,1)*255

        else:
            print(f"We don't know how to open that file... \n {self.spines_fn}")
            return

        path_ref = pathlib.Path(self.spines_fn)
        ext = path_ref.suffix

        save_fn = self.spines_fn.replace(ext, f"_spines.tif")
        save_fn_max = self.spines_fn.replace(ext, f"_spines_max.png")

        print('Saving stack and maximum intensity projection')
        io.mimsave(save_fn, stack)
        io.imsave(save_fn_max, stack.max(0))

        return save_fn

if __name__ == '__main__':
    s = Spines()
    