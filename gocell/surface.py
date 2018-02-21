import io
import numpy as np


def get_pixel_map(shape, normalized=False):
    z = (np.array(shape) - 1. if normalized else np.ones(2))[Ellipsis, None, None]
    z[z == 0] = 1
    return np.indices(shape) / z


class Surface:

    def __init__(self, shape, model=None, offset=(0, 0), mask=None):
        self.offset = offset
        self.model  = np.zeros(shape) if model is None else model[offset[0] : offset[0] + shape[0], \
                                                                  offset[1] : offset[1] + shape[1]]
        if mask is not None and isinstance(mask, np.ndarray):
            self.mask = mask.reshape(self.model.shape).astype(bool)
        else:
            self.mask = np.ones(self.model.shape).astype(bool)
            if mask is not None:
                self.mask = np.array([mask(x) for x in self.ndindex()]).reshape(self.model.shape).astype(bool)
    
    @staticmethod
    def create_from_image(img):
        img_diff = img.max() - img.min()
        if img_diff == 0: img_diff = 1
        img = (img - img.min()).astype(float) / img_diff
        return Surface(img.shape, img)
    
    @staticmethod
    def create_from_file(filepath):
        img = io.imread(filepath)
        return Surface.create_from_image(img)
    
    def ndindex(self, only_masked=True):
        """Generates all `x` of this surface.
        """
        for x in ndindex(self.model.shape):
            if not only_masked or self.mask[x]:
                yield (x[0] / float(self.model.shape[0] - 1), x[1] / float(self.model.shape[1] - 1))
            
    def rasterize(self, x):
        """Returns the pixel location of this surface, which corresponds to the nearest neighbor of `x`.
        """
        return tuple(np.multiply(x, (self.model.shape[0] - 1, self.model.shape[1] - 1)).round().astype(int))
    
    def clamp(self, x):
        """Returns the pixel location of this surface, which is close-most to `x`.
        
        The components of `x` must be non-negative.
        """
        x0 = min(x[0], self.model.shape[0] - 1)
        x1 = min(x[1], self.model.shape[1] - 1)
        return (int(round(x0)), int(round(x1)))
    
    def __getitem__(self, x):
        """Reads the pixel at `x` from the underlying matrix (nearest neighbor).
        
        The components of `x` must be between 0 and 1.
        """
        return self.model[self.rasterize(x)]
    
    def __setitem__(self, x, value):
        """Sets the pixel, which is close-most to `x` in the underlying matrix, to `value`.
        
        The components of `x` must be between 0 and 1.
        """
        self.model[self.rasterize(x)] = value
        
    def __len__(self):
        return self.model.size
    
    def get_map(self, normalized=True):
        return get_pixel_map(self.model.shape, normalized)

