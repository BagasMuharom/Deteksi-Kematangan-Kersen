import cv2 as cv
import numpy as np

class CitraContour:

    binary = None

    color = None
    
    labeled = None

    areaMin = 0

    areaMax = 0

    contours = []

    cropped = []

    classify = False

    classifier = None

    croppedResize = (0, 0)

    def __init__(self, binary, color):
        self.binary = binary
        self.color = color

    def findContour(self):
        # Mencari kontur
        contours = cv.findContours(self.binary.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        color = self.color.copy()
        self.labeled = self.color
        
        # loop over the contours
        for (i, c) in enumerate(contours):

            # Mengetahui lokasi kontur
            x,y,w,h = cv.boundingRect(c)
            
            if w*h < self.areaMin or w*h >= self.areaMax:
                continue
            
            # Menggambar garis pada gambar yang akan dilabeli
            cv.drawContours(self.labeled, [c], 0, (0, 255, 0), 3)

            crop = self.crop(color, c)

            if self.croppedResize is not (0, 0):
                crop = cv.resize(crop, self.croppedResize)

            if self.classify == True:
                self.classifyObject(image, crop, c)
            
            self.cropped.append(crop)

    def crop(self, color, c):
        # Mengetahui lokasi kontur
        x,y,w,h = cv.boundingRect(c)

        # Membuat gambar kosong sesuai ukuran asli
        canvas = np.zeros_like(color)

        # Menggambar objek kontur pada gambar kosong
        cv.drawContours(canvas, [c], 0, (255, 255, 255), -1)

        # Melakukan masking
        masked = cv.bitwise_and(color, canvas)

        # Memotong hasil masking sesuai ukuran kontur
        crop = masked[y:y+h, x:x+w]

        return crop

    def classifyObject(self, color, image, contour):
        # Mengetahui lokasi kontur
        x,y,w,h = cv.boundingRect(c)

        # Melakukan klasifikasi
        output = self.classifier.predict(image)

        # Menulis hasil klasifikasi
        cv.putText(self.labeled, "#{}".format(output), (int(x) - 10, int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)