import matplotlib.pyplot as plt
from CNNModel import CNNModel
from NNModel import NNModel
from Citra import Citra
import cv2 as cv

cap = cv.VideoCapture('data/data mentah/baru/video/2.3gp')
# cap = cv.VideoCapture('http://10.244.137.150:8080/video')
model = CNNModel('hasil learning/model CNN 99.0625.h5', 'categorical_crossentropy', 'sgd')
# model = NNModel('hasil learning/Model NN 94.375.h5', 'categorical_crossentropy', 'sgd')
model.outputClass = ['mentah', 'setang', 'hampir', 'matang']
    
while(True):
    ret, frame = cap.read()
    
    if ret == False:
        cap.set(2, 0)
        continue

    citra = Citra(frame)
    citra.resize((768, 576))
    citra.toGray()
    citra.toBinaryInv(150, 255)
    cc = citra.getContour(2000, 50000)
    cc.classify = True
    cc.classifier = model

    cc.croppedResize = (128, 128)
    cc.findContour()
    cv.imshow('Hasil', cc.labeled)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()