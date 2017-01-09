import utils
import math
import numpy as np
import constant
import waveform
import detector
from scipy import signal

class Analyse:
    def __init__(self, det = None):
        self.det = det
        

    def switchsignaldirection(self,wf):
        mean = self.getmean(wf)
#        print mean
        #newamp = (wf.amp-mean)
        newamp = mean - (wf.amp-mean)
        newwf = waveform.Waveform(wf.time,newamp)
        return newwf
        
    def producepowerwaveform(self, wf):
        vfeamp = utils.adctov_board(wf.amp)
        fewf = waveform.Waveform(wf.time,vfeamp,'an_fe')
        if self.det.type == 'norsat' or self.det.type=='helix':
#            fewf = self.switchsignaldirection(fewf)
            powerdet = (vfeamp-constant.boardoffsetGD)/constant.boardslopeGD
        else:
            powerdet = (vfeamp-self.det.board_k)/self.det.board_slope
#        vfeamp = fewf.amp
#        powerdet = (vfeamp+8)/4
        #        pdbm = (powerdet - self.det.m2_offset)/self.det.m2_slope
        pdbm = (powerdet - self.det.m3_offset)/self.det.m3_slope
        watt = utils.dbmtowatt(pdbm)
        newwf = waveform.Waveform(wf.time,watt,'an_watt')
#        newwf = waveform.Waveform(wf.time,vfeamp,'an_watt')
#        newwf = waveform.Waveform(wf.time,powerdet,'an_watt')
        return newwf
        
    def producesigmawaveform(self, wf):
        size = len(wf.amp)
        mean = np.mean(wf.amp[size/2:size*0.8])
        std = np.std(wf.amp[size/2:size*0.8])
        sigma = (wf.amp - mean)/std
        newwf = waveform.Waveform(wf.time,sigma,'an_sigma')
        return newwf

    def getsigma(self, wf):
        size = len(wf.amp)
        std = np.std(wf.amp[size/2:size*0.8])
        return std

    def getmax(self, wf):
        max = np.max(wf.amp)
        return max

    def getmean(self, wf):
        size = len(wf.amp)
        mean = np.mean(wf.amp[size/2:size*0.8])
        return mean

    def producemeanwaveform(self, wf):
        mean = self.getmean(wf)
        mean = (wf.amp)/mean
        newwf = waveform.Waveform(wf.time,mean,'an_mean')
        return newwf


    def lowpass(self, wf, fcut, order):
        filtamp = utils.lowpass(wf.amp,wf.sampling,order,fcut)
        newwf = waveform.Waveform(wf.time,filtamp,'an_filt')
        return newwf

    def lowpasshard(self, wf, fcut):
        filtamp = utils.lowpasshard(wf.amp,wf.sampling,fcut)
        newwf = waveform.Waveform(wf.time,filtamp,'an_filt')
        return newwf

    def crosscorrel(self, wf, envelopewf):
        newenv = np.interp(wf.time, envelopewf.time, envelopewf.amp/np.max(envelopewf.amp))
        crosscorrel = signal.correlate(wf.amp, newenv, mode='full')
        #here we cut the correlated waveform. 
        # it is not really clean now, but we select the window of the input size with the largest integral
        size = len(wf.time)
        halfccsize = int(len(crosscorrel)/2)
        goodindexstart = 0
        goodindexstop = 0
        maxint = -1.
        for i in range(halfccsize):
            indexstart = i
            indexstop = i+size
            newc = crosscorrel[indexstart:indexstop]
            if np.sum(newc) > maxint:
                maxint = np.sum(newc)
                goodindexstart = indexstart
                goodindexstop = indexstop
        newc = crosscorrel[goodindexstart:goodindexstop]
        newwf = waveform.Waveform(wf.time,newc,'an_correl')
        return newwf
