########################################
## produce a noise AND SIGNAL waveform #
## according  the detector type and the#
## method  given in argument          ##
## (defaut is norsat) method 3        ##
## The signal has a gaussian enveloppe##
########################################
from mod import *

parser = argparse.ArgumentParser()
parser.add_argument("--det", type=str, nargs='?',default='norsat', help="type of detector: gi, dmx, norsat, helix")
parser.add_argument("--method", type=int, nargs='?', default=3, help="power detection simulation  method: 1,2 or 3")
parser.add_argument("--snr", type=float, nargs='?', default=1, help="signal to noise ratio (noise=<noise>)")
parser.add_argument("--width", type=float, nargs='?', default=100, help="2sigma of the gaussian [ns]")
args = parser.parse_args()

print '#####################################'
print '###### detector: ', args.det ,' ######'
print '###### power det method: ', args.method ,' ######'
print '#####################################'


dettype = args.det
method = args.method

signalsnr = args.snr
signallength = args.width*1e-9
temp = 100
det = detector.Detector(temp=temp,type=dettype)
det.loadspectrum()
a_snr = np.linspace(0.1,2,10)
a_col = ['b','g','r','k']
#a_snr = np.linspace(0.1,1,10)
a_width = [10e-9,100e-9,500e-9, 1e-10]
#a_width = [10e-9,50e-9,100e-9,500e-9,1e-10]
iter = 1
fig = plt.figure(figsize=(12,6))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
snr = 20
for w,col in zip(a_width,a_col):
    for i in range(iter):
        sim = simulation.Simulation(snr =snr, siglength=w, det=det,sampling=5e9)
        sim.producetime()
        sim.producenoise(True)
        sim.setpowerenvelope('gauss')
        sim.producesignal()
        simwf = waveform.Waveform(sim.time,sim.noise+sim.signal, type='hf')
        wf = det.producesimwaveform(simwf,'adc',method)
        analyser = analyse.Analyse(det)
        powerwf = analyser.producepowerwaveform(wf)
        sigmawf = analyser.producesigmawaveform(powerwf)
        fft = np.fft.rfft(sigmawf.amp)
        fftfreq = np.fft.rfftfreq(len(sigmawf.amp),25e-9)
        if w < 1e-9:
            lab = 'noise'
        else:
            lab = str(w/1e-9)+ ' ns'
        ax1.plot(sigmawf.time/1e-6,sigmawf.amp,col,label=lab)
        ax2.loglog(fftfreq/1e6,np.absolute(fft),col,label=lab)

        
ax1.set_xlabel('time [us]')
ax1.set_ylabel('power [sigma]')

ax2.set_xlabel('frequency [MHz]')
ax2.set_ylabel('power [a.u.]')

#plt.ylabel('detection efficiency')
plt.legend(loc=1)
plt.show()
