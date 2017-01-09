#import utils
import math
import numpy as np
#import constant
import shower
import antenna
class Event:
    def __init__(self, fname = '', type=''):
        self.id = 0
        self.fname = fname
        self.shower = shower.Shower()
        self.antennas = []
        
    def loadevent(self):
        f = open(self.fname,'r')
        eventbool = False
        showerbool = False
        channelbool = False
        tracebool = False
        s = shower.Shower()
        antennas = []
        for l in f:
#            print l
            if 'event' in l:
                eventbool = True
                continue
            if eventbool == True:
                self.id = int(l.split()[0])
                eventbool = False
            if 'shower' in l:
                showerbool = True
                continue
            if showerbool == True:
                lshower = l.split()
                s.energy = float(lshower[0])
                s.x = float(lshower[1])
                s.y = float(lshower[2])
                s.z = float(lshower[3])
                s.theta = float(lshower[4])
                s.phi = float(lshower[5])
                showerbool = False
            if 'depth' in l:
                ldepth = l.split() 
                s.depth = np.asarray(ldepth[1:],float)
            if 'dedx' in l:
                ldedx = l.split() 
                s.dedx = np.asarray(ldedx[1:],float)
            if 'electron' in l:
                lelec = l.split() 
                s.electron = np.asarray(lelec[1:],float)
            if 'channel' in l:
                channelbool = True
                continue
            if channelbool == True and tracebool == False:
                lchan = l.split()
                ant = antenna.Antenna()
                ant.id = int(lchan[0])
                ant.starttime = int(lchan[1])
                ant.binwidth = float(lchan[2])
                tracebool = True
            if channelbool == True and tracebool == True:
                if 'powertrace' in l:
                    lpowertrace = l.split()
                    ant.power = np.asarray(lpowertrace[1:],float)
                    antennas.append(ant)
                    channelbool = False
                    tracebool = False
                    
        self.shower = s
        self.antennas = antennas


