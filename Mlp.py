import math
import numpy
from keras.models import Sequential
from keras.layers import Dense

class Mlp:

    model = None

    def __init__(self):
        self.model = Sequential()
        this.initNetwork()

    