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
print '###### snr: ', args.snr , ' gaussian width = ' ,args.width ,' ######'
print '#####################################'
dettype = args.det
method = args.method

signalsnr = args.snr
signallength = args.width*1e-9

det = detector.Detector(type=dettype)
det.loadspectrum()
sim = simulation.Simulation(snr =signalsnr, siglength=signallength, det=det,sampling=5e9)
sim.producetime()
sim.producenoise(True)
sim.setpowerenvelope('gauss')
sim.producesignal()
simwf = waveform.Waveform(sim.time,sim.noise+sim.signal, type='hf')
wf = det.producesimwaveform(simwf,'adc',method)
analyser = analyse.Analyse(det)
powerwf = analyser.producepowerwaveform(wf)
meanwf = analyser.producemeanwaveform(powerwf)
sigmawf = analyser.producesigmawaveform(powerwf)

filt1 = analyser.lowpass(meanwf,1e6,4)
filt1 = analyser.producesigmawaveform(filt1)
envelopewf = waveform.Waveform(sim.time,sim.powerenvelope)
cc = analyser.crosscorrel(meanwf,envelopewf)
cc = analyser.producesigmawaveform(cc)

#plt.plot(wf.time, wf.amp)
#plt.plot(powerwf.time, powerwf.amp)
fig = plt.figure(figsize=(15,10))
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.plot(wf.time, wf.amp)
ax2.plot(meanwf.time, meanwf.amp)
ax3.plot(sigmawf.time, sigmawf.amp)
ax3.plot(filt1.time, filt1.amp)
ax3.plot(cc.time, cc.amp)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()
