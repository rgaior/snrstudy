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
#a_snr = np.linspace(0.1,1,10)
a_width = [25e-9,50e-9,100e-9,500e-9]
#a_width = [10e-9,50e-9,100e-9,500e-9]
a_col = ['b','g','r','c']
iter = 100
for w,c in zip(a_width,a_col):
    a_eff = np.array([])
    a_matchedeff = np.array([])
    for snr in a_snr:
        detected = 0
        detectedmatched = 0
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
            meanwf = analyser.producemeanwaveform(powerwf)
            envelopewf = waveform.Waveform(sim.time,sim.powerenvelope)
            cc = analyser.crosscorrel(meanwf,envelopewf)
            cc = analyser.producesigmawaveform(cc)

            deltat = 1.875e-6
#            deltat = w
            truetimeofmax = sim.sigtime
            if (truetimeofmax - deltat <= sigmawf.gettimeofmax()  <= truetimeofmax + deltat ) and (sigmawf.getmax() > 7):
                detected+=1
            if (truetimeofmax - deltat <= cc.gettimeofmax()  <= truetimeofmax + deltat ) and (cc.getmax() > 7):
                detectedmatched+=1
        a_eff = np.append(a_eff,float(detected)/iter)
        a_matchedeff = np.append(a_matchedeff,float(detectedmatched)/iter)
    plt.plot(a_snr,a_eff*100,c+'.--')
    plt.plot(a_snr,a_matchedeff*100,c+'.-',lw=2,label=str(w/1e-9) + ' ns')
plt.xlabel('SNR')
plt.ylabel('detection efficiency')
plt.legend(loc=2)
plt.show()
