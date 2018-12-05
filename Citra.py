import cv2 as cv
import numpy as np
from CitraContour import CitraContour

class Citra:

    img = None

    processed = None

    def __init__(self, img):
        self.img = img
        self.processed = img

    def toGray(self):
        self.processed = cv.cvtColor(self.processed, cv.COLOR_BGR2GRAY)

    def toBinaryInv(self, tr_down, tr_up):
        self.processed = cv.threshold(self.processed, tr_down, tr_up, cv.THRESH_BINARY_INV)[1]
        
    def resize(self, size):
        self.img = cv.resize(self.img, size)
        self.processed = cv.resize(self.processed, size)

    def getContour(self, areaMin, areaMax):
        contour = CitraContour(self.processed, self.img)
        contour.areaMin = areaMin
        contour.areaMax = areaMax

        return contour
    
    def getHistogram(self):
        colors = ('r','g','b')

        hist = np.array([])

        for i, color in enumerate(colors):
            tmp = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
            histr = cv.calcHist([tmp], [i], None, [256], [0, 256])
            hist = np.concatenate((hist, histr.flatten().astype(int)), axis = None)
            
        return hist.astype(int)

