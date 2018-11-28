import cv2 as cv
import numpy as np

class Contour:

    contour = None

    def __init__(self, contour):
        self.contour = contour