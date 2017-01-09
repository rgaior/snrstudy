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
parser.add_argument("--tsys", type=float, nargs='?', default=50, help="system noise temperture (integer)")
parser.add_argument("--folder", type=str, nargs='?', default=constant.simdatafolder, help="folder where the event in text format are located")

args = parser.parse_args()
print '#####################################'
print '###### detector: ', args.det ,' ######'
print '###### power det method: ', args.method ,' ######'
print '###### system temp: ', args.tsys ,' #########'
print '###### data folder ', args.folder ,' #########'
print '#####################################'

dettype = args.det
method = args.method
tsys = args.tsys
datafolder = args.folder

det = detector.Detector(type=dettype,temp=tsys)
det.loadspectrum()

fig = plt.figure(figsize=(12,6))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

names = glob.glob(datafolder + '*')
for n in names:
    ev = event.Event(fname=n, type='test')
    ev.loadevent()
    for ant in ev.antennas:
        time = ant.maketimearray()
        power = ant.power
        if np.max(power)< 5e-14:
            continue
#        print np.max(power)
        print n
        sim = simulation.Simulation(det=det,sampling=5e9)
        sim.producetime()
        sim.producenoise(True)
        sim.setpowerenvelopewitharray([time,power])
        sim.producesignal()
        simwf = waveform.Waveform(sim.time,sim.noise+sim.signal, type='hf')
        
        wf = det.producesimwaveform(simwf,'adc',method)
        analyser = analyse.Analyse(det)
        powerwf = analyser.producepowerwaveform(wf)
        meanwf = analyser.producemeanwaveform(powerwf)
        sigmawf = analyser.producesigmawaveform(powerwf)
        envelopewf = waveform.Waveform(sim.time,sim.powerenvelope)
        cc = analyser.crosscorrel(meanwf,envelopewf)
        cc = analyser.producesigmawaveform(cc)        

        ax1.plot(sigmawf.amp)
        ax2.plot(cc.amp)

ax1.set_xlim(50,700)
ax2.set_xlim(50,700)
ax1.set_ylim(-5,10)
ax2.set_ylim(-5,10)
#gmawf = analyser.producesigmawaveform(powerwf)

# filt1 = analyser.lowpass(meanwf,1e6,4)
# filt1 = analyser.producesigmawaveform(filt1)

# #plt.plot(wf.time, wf.amp)
# #plt.plot(powerwf.time, powerwf.amp)
# fig = plt.figure(figsize=(15,10))
# ax1 = plt.subplot(311)
# ax2 = plt.subplot(312)
# ax3 = plt.subplot(313)
# ax1.plot(wf.time, wf.amp)
# ax2.plot(meanwf.time, meanwf.amp)
# ax3.plot(sigmawf.time, sigmawf.amp)
# ax3.plot(filt1.time, filt1.amp)
# ax3.plot(cc.time, cc.amp)
# plt.xlabel('time')
# plt.ylabel('amplitude')
plt.show()
