import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

print("hello, world")
print("OpenCV version: ", cv2.__version__)
print("Numpy version: ", np.__version__)
