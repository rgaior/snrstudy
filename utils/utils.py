import numpy as np
from scipy import signal
import math
import constant
###############################################
####  reading functions                 #####
###############################################

# read a file with: time power [W]
def readsimfile(file):
    f = open(file,'r+')
    time = np.array([])
    power = np.array([])
    for l in f:
        lsplit = l.split()
        time = np.append(time,float(lsplit[0]))
        power = np.append(power,float(lsplit[1]))
    return [time,power]

def readscopefile(filename):
    f = open(filename,'r+')
    time = np.array([])
    v = np.array([])
    count = 0
    for l in f:
        if count < 6:
            count = count+1
            continue
        lsplit = l.split(',')
        time = np.append(time,float(lsplit[-2]))
        v = np.append(v,float(lsplit[-1]))
    return [time,v]

###############################################
####  producing functions                 #####
###############################################

def wf_normal(mean,sigma,nrofsamples):
    return np.random.normal(mean,sigma,nrofsamples)

def wf_dirac(nrofsamples):
    return  np.append(np.array([1]),np.zeros(nrofsamples-1))

def wf_sine(freq, amp, deltat, tend):
    t = np.arange(0, tend + deltat, deltat)
    thesine = amp*np.sin(2*math.pi*freq*t)
    return thesine


def func_normedgauss(x,mean,sigma):
    #a = (1./(sigma*( math.sqrt(2*math.pi) )) )*np.exp(-0.5* ((x - mean)/sigma)**2 )
    a = np.exp(-0.5* ((x - mean)/sigma)**2 )
    return a



###############################################
#### conversion function (voltage to adc, #####
#### voltage FE to voltage board etc... ) #####
###############################################
#for np array
#adc counts to volt at the FE input (between 0-1V)
def adctov_fe(adc):
    return adc.astype(float)/1024
def v_fetoadc(vfe):
    return vfe.astype(float)*1024

#voltage at front end to voltage at GIGAS/EASIER board
def v_fetov_board(vfe):
    return vfe*(-2)
def v_boardtov_fe(vboard):
    return vboard*(-1/2)

#adc to v board (between -2 and 0 V)
def adctov_board(adc):
    return v_fetov_board(adctov_fe(adc))
def v_boardtoadc(vboard):
    return v_fetoadc(v_boardtov_fe(vboard))

#switch the direction of the signal w.r.t. baseline
#needed for the norsat/helix
def switchsignaldirection(amp):
    mean = np.mean

def dbmtowatt(dbm):
    return 10*np.power(10., (dbm - 30) /10)

def dbtowatt(db):
    return 10*np.power(10., db)

def watttodb(db):
    return 10*np.log10(db)

def watttodbm(watt):
    return 10*np.log10(watt) + 30



###############################################
####              filtering               #####
###############################################

def lowpass(amp, sampling, order, fcut):
    Nyfreq = sampling/2
    ratiofcut = float(fcut)/Nyfreq
    b, a = signal.butter(order, ratiofcut, 'low')
    filtered = signal.filtfilt(b, a, amp)
    return filtered

def lowpasshard(amp, sampling, fcut):
    fft = np.fft.rfft(amp)
    freq = np.fft.rfftfreq(len(fft),float(1./sampling))
    Nyfreq = sampling/2
    min = np.min(np.absolute(fft))
    ratiofcut = float(fcut)/Nyfreq
    size = len(fft)
    newpass = fft[:int(ratiofcut*size)]
    sizeofzeros = size - len(newpass)
    newcut = np.zeros(sizeofzeros)
    newfft = np.append(newpass,newcut)
    out = np.fft.irfft(newfft)
    return out.real

def highpass(amp, sampling, order, fcut):
    Nyfreq = sampling/2
    ratiofcut = float(fcut)/Nyfreq
    b, a = signal.butter(order, ratiofcut, 'high')
    filtered = signal.filtfilt(b, a, amp)
    return filtered


def slidingwindow(y,bins,option=None):                                          
    window = np.ones(bins)/bins                                                 
    if option is not None:                                                      
        if option.lower() not in ['full','same','valid']:                       
            print 'invalid option, check your sliding window'                   
    if option == None:                                                          
        return np.convolve(y,window,'same')                                     
    else:                                                                       
        return np.convolve(y,window,option) 

def rms(x):
    return np.sqrt(np.mean(x**2))

#################################
### simulation function   #######
#################################
def produceresponse(time,amp,gain,tau):
    tend = 500e-9
    period = time[1] - time[0]
    x = np.arange(0,tend,period)
    convfunc = period*np.exp(-x/tau)/( -(math.exp(-tend/tau) - 1)*tau)
    power = gain*(amp**2)/50
    signal = 10*np.log10(power) + 30
    resp = np.convolve(signal,convfunc,'valid')
    newtime = np.linspace(time[0], time[0]+ float(len(resp))*period, len(resp))
    newamp = resp
    return [newtime,newamp]

def produceresponse2(time,amp,gain,tau):
    tend = time[-1] - time[0]
    period = time[1] - time[0]
    x = np.arange(0,tend+period,period)
    convfunc = period*np.exp(-x/tau)/( -(math.exp(-tend/tau) - 1)*tau)
    fft = np.fft.rfft(convfunc)
    power = gain*(amp**2)/50
    signal = 10*np.log10(power) + 30
    fftsig = np.fft.rfft(signal)
    out = fft*fftsig
    out = np.fft.irfft(out)
    return [time,out]


def deconv(time,amp,gain,tau):
    tend = time[-1] - time[0]
    period = time[1] - time[0]
    x = np.arange(0,tend+period,period)
    convfunc = period*np.exp(-x/tau)/( -(math.exp(-tend/tau) - 1)*tau)
    fftconv = np.fft.rfft(convfunc)
    fftsig = np.fft.rfft(amp)
    out = fftsig/fftconv
    out = np.fft.irfft(out)
    return [time[:-1],out]

def powerdetfunc2(x,a, k, j):
    return a*np.exp(-(k*x)) + j

def m2_powerdetectorsim(time,amp,gain,capaornot):
    logamp = watttodbm(gain*amp*amp/50) 
    freq = np.fft.rfftfreq(len(time),time[1]-time[0])
    fft = np.fft.rfft(logamp)
    # no capa
    if capaornot == 1: 
        file = constant.c2_file
        phase = np.load(file)['phase']
        freqori = np.load(file)['freq']
        interpphase = np.interp(freq,freqori,phase)
        prefact = constant.c2_prefact
        k = constant.c2_k
        j = constant.c2_j
        spec = powerdetfunc2(freq[1:],prefact,k,j)
        dcval = np.absolute(constant.c2_slope + constant.c2_offset/np.mean(logamp))
        spec = np.insert(spec,0,dcval)
        response = spec*np.exp(1j*interpphase)
        outfft = fft*response
        out = np.fft.irfft(outfft)
    if capaornot == 0: 
        file = constant.nc2_file
        phase = np.load(file)['phase']
        freqori = np.load(file)['freq']
        interpphase = np.interp(freq,freqori,phase)
        prefact = constant.nc2_prefact
        k = constant.nc2_k
        j = constant.nc2_j
        spec = powerdetfunc2(freq[1:],prefact,k,j)
        dcval = np.absolute(constant.nc2_slope + constant.nc2_offset/np.mean(logamp))
        spec = np.insert(spec,0,dcval)
        response = spec*np.exp(1j*interpphase)
        outfft = fft*response
        out = np.fft.irfft(outfft)
    return out

def m3_powerdetectorsim(time,amp,gain,tau,slope,offset):
    size = len(time)
    conv = produceresponse2(time,amp,gain,tau)
    sim = conv[1] 
    polyconv_pd = np.poly1d([slope,offset])
    simpd = polyconv_pd(sim)
    return [conv[0],simpd]

def findparam_3(wfRF,wfPD,t):
    size = len(wfRF[1])
    conv = produceresponse(wfRF[0],wfRF[1],t)
    real = wfPD[1] -  getbaseline(wfPD[1],1)
    sim = conv[1] - getbaseline(conv[1],1)
    #resize the two waveforms to the same size (because of the convolution)                                                                 
    [real,sim] = resize(real,sim)
    time = gettime(wfPD[0],conv[0])
    delay = finddelay2(real,sim)
    simshifted =  np.roll(sim,delay)
    #fit the conv vs power:                                                                                                                 
    fitconv_pd = np.polyfit(simshifted,real,1)
    #polyconv_pd = np.poly1d(fitconv_pd)
    polyconv_pd = np.poly1d([-0.0252,0])
    simpd = polyconv_pd(simshifted)
    return [[time,simpd],[time,real]]


def boardspecfunc(freq,prefact,mu,sigma,k):
    return  prefact*np.exp(-(freq/1e6 - mu)**2/(2*sigma**2)) + k
def boardphasefunc(freq,a,b,c):
    pgainphase  = np.poly1d([a,b,c])
    phase = pgainphase(freq/1e6)
    return phase
    

def resize(amp1,amp2):
    difflen = len(amp1) - len(amp2)
    if difflen == 0:
        return [amp1,amp2]
    elif difflen == 1:
        amp1 = amp1[1:]
    elif difflen == -1:
        amp2 = amp2[:-1]
    elif difflen %2 == 0  and difflen > 0:
        amp1 = amp1[difflen/2:-difflen/2]
    elif difflen %2 == 0  and difflen < 0:
        diff = difflen/2
        amp2 = amp2[diff:-diff]
    elif difflen %2 != 0  and difflen < 0:
        diff = int(np.absolute(float(difflen)/2))
        amp2 = amp2[diff+1:-diff]
    elif difflen %2 != 0  and difflen > 0:
        diff = int(np.absolute(difflen/2))
        amp1 = amp1[diff+1:-diff]
    return [amp1,amp2]


def gettime(time1,time2):
    difflen = len(time1) - len(time2)
    if difflen==1:
        return [time1[:-1],time2]
    elif difflen==-1:
        return [time1,time2[:-1]]
    elif len(time1) > len(time2):
        time1 = time1[difflen/2:-difflen/2]
    elif len(time1) < len(time2):
        time2 = time1[-difflen/2:difflen/2]
    return [time1,time2]

def finddelay2(amp1,amp2):
    fftamp1 = np.fft.fft(amp1)
    fftamp2 = np.fft.fft(amp2)
    cfftamp1 = -fftamp1.conjugate()
    cfftamp2 = -fftamp2.conjugate()
    return np.argmax(np.abs(np.fft.ifft(fftamp1*cfftamp2)))

def alignwaveform(amp1,amp2,pos):
    if pos == True:
        delay = np.argmax(amp1) - np.argmax(amp2)
    else:
        delay = np.argmin(amp1) - np.argmin(amp2)
#    print delay
    amp2 = np.roll(amp2,delay)
    return [amp1,amp2]

def alignwaveform2(amp1,amp2):
    [amp1,amp2] = resize(amp1,amp2)
    delay = finddelay2(amp1,amp2)
    amp2 = np.roll(amp2,delay)
    return [amp1,amp2]

def resample(time, amp, newsampling):
    newtime = np.arange(time[0],time[-1],1/newsampling)
    newamp = np.interp(newtime,time,amp)
    return [newtime,newamp]

def getbaseline(amp, portion):
    size= len(amp)
    return np.mean(amp[:int(size*portion)] )


def linearize(amp):
    amp = adctov_board(amp)
    amp_pd = (amp - constant.boardoffset)/(constant.boardslope)
    power_dbm = (amp_pd - constant.c3_powerdetoffset)/(constant.c3_powerdetslope)
    power_watt = dbmtowatt(power_dbm) 
    return power_watt

def normalize(amp):
    return (amp - np.mean(amp))/np.std(amp)


def getdistance(p1,p2):
    return np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2  )
