#import utils
import math
import numpy as np
#import waveform
#import constant
class Antenna:
    def __init__(self):
        self.id = 0
        self.starttime = 0
        self.binwidth = 0
        self.power = np.array([])
        self.maxenvelope = None
        self.envelope = None
        self.adcwf_power = None
        self.adcwf_sigma = None
        self.filterwf_sigma = None

    def maketimearray(self):
        return np.arange(self.starttime,self.binwidth*len(self.power),self.binwidth)


