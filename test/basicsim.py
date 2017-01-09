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
temp = 100
det = detector.Detector(temp=temp,type=dettype)
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
sigmawf = analyser.producesigmawaveform(powerwf)

print len(sim.time), ' ' , len(sim.powerenvelope) 
fig= plt.figure(figsize=(10,8))
fig.suptitle('gaussian input /' + ' width = '+ str(signallength/1e-9) +'ns / SNR = ' +str(signalsnr),fontsize=15,fontweight='bold')
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
#plt.plot(wf.time, wf.amp)
ax1.plot(wf.time/1e-6, wf.amp)
#ax2.plot(sigmawf.time/1e-6, powerwf.amp)
ax2.plot(sigmawf.time/1e-6, sigmawf.amp)
#ax.plot(sim.time, sim.signal,lw=2)
#plt.plot(sim.time/1e-6, sim.powerenvelope,lw=2)
#plt.plot(simwf.time/1e-6, simwf.amp,lw=2)
#ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
ax2.set_xlabel('time [us]')
ax2.set_ylabel('power [sigma]')
ax1.set_ylabel('power [ADC]')
#ax.set_ylabel('power')
plt.show()
