import utils
import math
import numpy as np
from scipy import signal
import constant
import waveform

#front end filter cut frquency
fcut = constant.fefcut
fesampling = constant.fesampling

class Detector:
    def __init__(self, temp=100, gain=1e6, f1=0.95e9,f2=1.750e9, tau=5e-9,type=''):
        self.temp = temp
        self.gain = gain  
        self.f1 = f1
        self.f2 = f2
        self.tau = tau
        self.noisespectrum = []
        self.pnoise = self.setpnoise()

        self.m1_slope = 0
        self.m1_offset = 0
        self.m1_tau = 0

        self.m2_slope = 0
        self.m2_offset = 0
        self.m2_prefact = 0
        self.m2_k = 0 
        self.m2_j = 0
        self.m2_dc = 0 

        self.m3_slope = 0
        self.m3_offset = 0
        self.m3_tau = 0

        self.capaornot = 0
        self.type = type
        # cf new note on detector
        if type == 'dmx' or type=='norsat' or type=='helix' or type=='':
            self.capaornot = 0
            #method 1/with capacitor  power detector simulation constant
            self.m1_offset = constant.nc_powerdetoffset
            self.m1_slope = constant.nc_powerdetslope
            self.m1_tau = constant.nc_powerdettau
            #method 2/with capacitor  power detector simulation constant
            self.m2_slope = constant.nc2_slope
            self.m2_offset = constant.nc2_offset
            self.m2_prefact = constant.nc2_prefact
            self.m2_k = constant.nc2_k
            self.m2_j = constant.nc2_j
            self.m2_dc = constant.nc2_dc
            #method 3/with capacitor  power detector simulation constant
            self.m3_slope = constant.nc3_powerdetslope
            self.m3_offset = constant.nc3_powerdetoffset
            self.m3_tau = constant.nc3_powerdettau
        elif type=='gi':
            self.capaornot = 1
            #method 1/no capacitor  power detector simulation constant
            self.m1_offset = constant.c_powerdetoffset
            self.m1_slope = constant.c_powerdetslope
            self.m1_tau = constant.c_powerdettau
            #method 2/no capacitor  power detector simulation constant
            self.m2_slope = constant.c2_slope
            self.m2_offset = constant.c2_offset
            self.m2_prefact = constant.c2_prefact
            self.m2_k = constant.c2_k
            self.m2_j = constant.c2_j
            self.m2_dc = constant.c2_dc
            #method 3/no capacitor  power detector simulation constant
            self.m3_slope = constant.c3_powerdetslope
            self.m3_offset = constant.c3_powerdetoffset
            self.m3_tau = constant.c3_powerdettau
        #DC caracteristics of EASIER Board
        self.board_k = constant.boardoffset
        self.board_slope = constant.boardslope

    # sets the total noise power for the given temperature and spectrum (no gain included)
    def setpnoise(self):
        if len(self.noisespectrum) != 0:
            deltaB = np.absolute(np.trapz(self.noisespectrum[1],self.noisespectrum[0]))
            self.pnoise = constant.kb*self.temp*deltaB
        elif len(self.noisespectrum) == 0:
            self.pnoise = constant.kb*self.temp*(self.f2 - self.f1)

        
    #return [time, amp] for the different stages of the detector
    def producesimwaveform(self, simwf, stage, method=None):
        stagelc = stage.lower()
        if stagelc not in ['logresponse','powerdetector','board','fefilter','timesampled','adc']:
            print 'choose among these stages: \n ', 'logresponse or powerdetector or board or fefilter or timesampled or adc'
            return 
        elif stagelc == 'logresponse':
            return self.produceresponse(simwf)
        elif stagelc == 'powerdetector':
            if method == None or method ==1:
                afterpd = self.m1_powerdetsim(simwf)
            elif method==2:
                afterpd = self.m2_powerdetsim(simwf)
            elif method==3:
                afterpd = self.m3_powerdetsim(simwf)            
            return afterpd
        elif stagelc == 'board':
            if method == None or method ==1:
                afterpd = self.m1_powerdetsim(simwf)
            elif method==2:
                afterpd = self.m2_powerdetsim(simwf)
            elif method==3:
                afterpd = self.m3_powerdetsim(simwf)
            return self.adaptationboard2(afterpd)
        elif stagelc == 'fefilter':
            if method == None or method ==1:
                afterpd = self.m1_powerdetsim(simwf)
            elif method==2:
                afterpd = self.m2_powerdetsim(simwf)
            elif method==3:
                afterpd = self.m3_powerdetsim(simwf)
            afterboard = self.adaptationboard2(afterpd)
            return self.FEfilter(afterboard)
        elif stagelc == 'timesampled':
            if method == None or method ==1:
                afterpd = self.m1_powerdetsim(simwf)
            elif method==2:
                afterpd = self.m2_powerdetsim(simwf)
            elif method==3:
                afterpd = self.m3_powerdetsim(simwf)
            afterboard = self.adaptationboard2(afterpd)
            afterFEfilter = self.FEfilter(afterboard)
            return self.FEtimesampling(afterFEfilter)
        elif stagelc == 'adc':
            if method == None or method ==1:
                afterpd = self.m1_powerdetsim(simwf)
            elif method==2:
                afterpd = self.m2_powerdetsim(simwf)
            elif method==3:
                afterpd = self.m3_powerdetsim(simwf)
            afterboard = self.adaptationboard2(afterpd)
            afterFEfilter = self.FEfilter(afterboard)
            timesampled = self.FEtimesampling(afterFEfilter)
            return self.FEampsampling(timesampled)

    def produceresponse(self,wf):
        tend = 500e-9
        period = 1./wf.sampling
        x = np.arange(0,tend,period)
        convfunc = period*np.exp(-x/self.m1_tau)/( -(math.exp(-tend/self.m1_tau) - 1)*self.m1_tau)
        # response in dBm
        power = self.gain*(wf.amp*wf.amp)/constant.impedance
        signal = 10*np.log10(power) + 30
        resp = np.convolve(signal,convfunc,'valid')
        newtime = np.linspace(wf.time[0], float(len(resp))/wf.sampling+wf.time[0], len(resp))
        newamp = resp        
        newwf = waveform.Waveform(newtime,newamp,'logresponse')
        return newwf

#power detector characteristic (P[dBm] vs V_pd[V])
    def powerdetlinear(self, wf):
        newwf = waveform.Waveform(wf.time,self.m1_offset + self.m1_slope*wf.amp,'powerdetector')
        return newwf

    #method #1:
    # parameter are found by comparing HF and power detector data recorded on oscilloscope.
    # we do a convolution of the HF in dBm and then fit the time constant, and the caracteristics (V_pd = a*P + b)
    def m1_powerdetsim(self,wf):
        resp = self.produceresponse(wf)
        newwf = self.powerdetlinear(resp)
        return newwf

    #method #2:
    # the response is found by setting the caracteristic V_pd = a*P + b for DC component.
    # then we find the response in frequency of the power detector.
    # to simulate the output signal we multiply the FFT of the input by the response found before.
    def m2_powerdetsim(self, wf):
        newamp = utils.m2_powerdetectorsim(wf.time,wf.amp,self.gain,self.capaornot)
        newwf = waveform.Waveform(wf.time,newamp,'powerdetector')
        return newwf

    #method #3:
    # similar to method #1 but the caracteristics V_pd = a*P + b is fixed with the DC parameters.
    # the time constant \tau is the only fitted parameter
    def m3_powerdetsim(self, wf):
        newwf = utils.m3_powerdetectorsim(wf.time,wf.amp,self.gain,self.m3_tau,self.m3_slope, self.m3_offset)
        newwf = waveform.Waveform(newwf[0],newwf[1],'powerdetector')
        return newwf

#adaptation board characteristic (V_pd [V] vs V_board [V])
    def adaptationboard(self, wf):
        newwf = waveform.Waveform(wf.time,self.board_k + self.board_slope*wf.amp,'board')
        return newwf

#simulation the easier board accounting for the spectrum of the amplifier
# (see calib/board_HFcarac.py for the numbers)
    def adaptationboard2(self, wf):
        # first we compute the fft of the waveform:
        fft = np.fft.rfft(wf.amp)
        spec = np.absolute(fft)
        fftfreq = np.fft.rfftfreq(len(wf.amp),wf.time[1] - wf.time[0])
        # then we produce the gain of the amplifier for the given frequencies (exept DC)
        # according the study that extracted the parameters empirically
        prefact = constant.boardspecprefact
        mu = constant.boardspecmu
        sigma = constant.boardspecsigma
        k = constant.boardspeck
        gainspec = utils.boardspecfunc(fftfreq,prefact,mu,sigma,k)
        a = constant.boardphasea
        b = constant.boardphasea
        c = constant.boardphasea
        gainphase = utils.boardphasefunc(fftfreq,a,b,c)
        if self.type=='norsat' or self.type=='helix':
            gainfft = gainspec[1:]*np.exp(1j*gainphase[1:])
#            dcgain = -self.board_slope - self.board_k/np.mean(wf.amp)
            #!!! hardcoded value for the board in the case of GD.
            dcgain = -self.board_slope - 8/np.mean(wf.amp)
        else:    
            gainfft = -gainspec[1:]*np.exp(1j*gainphase[1:])
            dcgain = self.board_slope + self.board_k/np.mean(wf.amp)
        gainfft = np.insert(gainfft,0,dcgain)
        newamp = np.fft.irfft(gainfft*fft)
        newwf = waveform.Waveform(wf.time[:-1],newamp,'board')
        return newwf

#simulate the Front end filter of Auger electronics 
    def FEfilter(self, wf):
        Nyfreq = wf.sampling/2
        ratiofreq = float(fcut)/Nyfreq
        b, a = signal.butter(4, ratiofreq)
        y = utils.lowpass(wf.amp,wf.sampling,4,1*fcut)
        newwf = waveform.Waveform(wf.time,y,'fefilter')
        return newwf

    # the sampling in time (every 25ns)
    def FEtimesampling(self, wf):
        #first time sampling:
        step = float(1./fesampling)
        tracelength = wf.length
        nrofpoints = int(tracelength/step)
        newtime = np.linspace(wf.tstart,wf.tend,nrofpoints)
        [a,b] = utils.resize(wf.time,wf.amp)
        newy = np.interp(newtime,a,b)
        newwf = waveform.Waveform(newtime,newy,'timesampled')
        return newwf

    # the gain of the input amplifier g = (-1/2)
    # the sampling in amplitude (0-1V to 0-1023ADC)
    def FEampsampling(self, wf):
        #first time sampling:
        newy = -0.5*wf.amp*1023
        newy = newy.astype(int)
        newwf = waveform.Waveform(wf.time,newy,'adc')
        return newwf

    #loads the normalised spectrum
    def loadspectrum(self):
        fname = ''
        if 'norsat' in self.type.lower():
            fname = 'Norsat8115_n.txt'
        if 'dmx' in self.type.lower():
            fname = 'DMX241_n.txt'
        if 'gi' in self.type.lower():
            fname = 'GI301_n.txt'
        if 'helix' in self.type.lower():
            fname = 'helix_n.txt'
        f = open(constant.spectrafolder+fname,'r')
        freq = np.array([])
        gain = np.array([])
        for l in f:
            freq = np.append(freq,float(l.split()[0]))
            gain = np.append(gain,float(l.split()[1]))
        self.noisespectrum = [freq,gain]
        self.setpnoise()
