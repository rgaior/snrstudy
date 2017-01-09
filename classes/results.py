import utils
import math
import numpy as np
import constant
import waveform
import detector
from scipy import signal

class Results:
    def __init__(self):
        self.ant = None
        self.shower = None
        self.evid = None
        self.y = None
        
    def setevid(self,evid):
        self.evid = evid
        
    
