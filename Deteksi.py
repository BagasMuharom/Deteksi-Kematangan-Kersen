import matplotlib.pyplot as plt
from Mlp import Mlp
from Citra import Citra
import cv2 as cv

# cap = cv.VideoCapture('http://192.168.13.7:8080/video')
cap = cv.VideoCapture('data/data mentah/VID_20181126_174054.mp4')
mlp = Mlp('hasil learning/model92.1875.json', 'hasil learning/model92.1875.h5', 'categorical_crossentropy', 'sgd')
mlp.outputClass = ['mentah', '2', '3', 'matang']
    
while(True):
    ret, frame = cap.read()
    
    if ret == False:
        cap.set(2, 0)
        continue

    citra = Citra(frame)
    citra.resize((750, 500))
    citra.toGray()
    citra.toBinaryInv(150, 255)
    cc = citra.getContour(9000, 50000)
    cc.classify = True
    cc.classifier = mlp
    cc.croppedResize = (128, 128)
    cc.findContour()
    cv.imshow('Hasil', cc.labeled)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()