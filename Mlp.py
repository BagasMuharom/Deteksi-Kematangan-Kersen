import math
import numpy
from keras.models import model_from_json

class Mlp:

    model = None

    loss = None

    optimizer = None

    outputClass = []

    def __init__(self, modelPath, weightPath, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.initNetwork(modelPath, weightPath)

    def initNetwork(self, modelPath, weightPath):
        model_json = open(modelPath, 'r')
        self.model = model_from_json(model_json.read())
        model_json.close()
        self.model.load_weights(weightPath)
        self.model.compile(loss = self.loss, optimizer= self.optimizer, metrics=['accuracy'])

    def predict(self, data):
        predicted = self.model.predict(numpy.array([data]))

        return self.outputClass[numpy.argmax(predicted[0])]

    