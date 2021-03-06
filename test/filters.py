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

f1 = 1e6
filt1 = analyser.lowpass(meanwf,f1,4)
filt1 = analyser.producesigmawaveform(filt1)

f2 = 2e6
filt2 = analyser.lowpass(meanwf,f2,4)
filt2 = analyser.producesigmawaveform(filt2)

f3 = 10e6
filt3 = analyser.lowpass(meanwf,f3,4)
filt3 = analyser.producesigmawaveform(filt3)


envelopewf = waveform.Waveform(sim.time,sim.powerenvelope)
cc = analyser.crosscorrel(meanwf,envelopewf)
cc = analyser.producesigmawaveform(cc)


fig =plt.figure()
fig.suptitle('gaussian input /' + ' width = '+ str(signallength/1e-9) +'ns / SNR = ' +str(signalsnr),fontsize=15,fontweight='bold')
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.plot(sigmawf.time/1e-6,sigmawf.amp,label= 'original')
#ax2.plot(filt1.time/1e-6,filt1.amp,lw=2,label='fcut = '+str(f1/1e6)+' MHz' )
#ax2.plot(filt2.time/1e-6,filt2.amp,lw=2,label='fcut = '+str(f2/1e6)+' MHz' )
#ax2.plot(filt3.time/1e-6,filt3.amp,lw=2,label='fcut = '+str(f3/1e6)+' MHz' )
ax2.plot(cc.time/1e-6,cc.amp,lw=2,label='matched filter' )
ax2.set_xlabel('time [us]')
ax1.set_ylabel('power [sigma]')
ax2.set_ylabel('power [sigma]')
plt.legend()
ax1.set_xlim(2,18)
ax2.set_xlim(2,18)
ax1.set_ylim(-4,15)
ax2.set_ylim(-4,15)

fig3 = plt.figure(figsize=(12,6))
plt.plot(sim.time/1e-6,sim.powerenvelope,lw=2)
plt.xlabel('time [us]')

fig4 = plt.figure(figsize=(12,6))
fftwf = np.fft.rfft(wf.amp)
fftenv = np.fft.rfft(sim.powerenvelope)
fftout = np.fft.rfft(cc.amp)
fftfreqwf = np.fft.rfftfreq(len(wf.amp),25e-9)
fftfreqenv = np.fft.rfftfreq(len(np.sqrt(sim.powerenvelope)),1/sim.sampling)
fftfreqout = np.fft.rfftfreq(len(cc.amp),25e-9)
plt.loglog(fftfreqwf, np.absolute(fftwf))
fig5 = plt.figure(figsize=(12,6))
plt.loglog(fftfreqenv, np.absolute(fftenv))
fig6 = plt.figure(figsize=(12,6))
plt.loglog(fftfreqout, np.absolute(fftout))
plt.show()
