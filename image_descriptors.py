from skimage import feature
import numpy as np


class LocalBinaryPatterns:

    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


class HOG:
    def __init__(self, orientations, pixels_per_cell, cpb, viz, multi):
        self.orientations = orientations
        self.ppc = pixels_per_cell
        self.cpb = cpb
        self.viz = viz
        self.multi = multi

    def describe(self, image, eps=1e-7):
        hog = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.ppc,
                          cells_per_block=self.cpb, block_norm='L2-Hys', visualize=self.viz, feature_vector=True)
        return np.array(hog)

