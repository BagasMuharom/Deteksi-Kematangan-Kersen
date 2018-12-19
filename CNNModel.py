import math
import numpy
from keras.models import load_model

class CNNModel:

    model = None

    loss = None

    optimizer = None

    outputClass = []

    def __init__(self, modelPath, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.initNetwork(modelPath)

    def initNetwork(self, modelPath):
        self.model = load_model(modelPath)
        self.model.compile(loss = self.loss, optimizer= self.optimizer, metrics=['accuracy'])

    def predict(self, data):
        predicted = self.model.predict(numpy.array([data]))

        return self.outputClass[numpy.argmax(predicted[0])] + ' ' + '{0:.2f}'.format(predicted[0, numpy.argmax(predicted[0])] * 100) + '%'

    