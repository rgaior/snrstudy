########################################
## produce a noise AND SIGNAL waveform #
## according  the detector type and the#
## method  given in argument          ##
## (defaut is norsat) method 3        ##
## The signal has a gaussian enveloppe##
########################################
from mod import *
from scipy.optimize import curve_fit
def func(x, a, sigma,mu,b):
#    return (a/sigma)*np.exp(-((x - mu)/sigma)**2) + b
    return a*np.exp(-((x - mu)/sigma)**2) + b

parser = argparse.ArgumentParser()
parser.add_argument("--det", type=str, nargs='?',default='norsat', help="type of detector: gi, dmx, norsat, helix")
parser.add_argument("--method", type=int, nargs='?', default=3, help="power detection simulation  method: 1,2 or 3")
args = parser.parse_args()

print '#####################################'
print '###### detector: ', args.det ,' ######'
print '###### power det method: ', args.method ,' ######'
print '#####################################'
dettype = args.det
method = args.method

signallength = 500*1e-9
temp = 100
det = detector.Detector(temp=temp,type=dettype)
det.loadspectrum()
a_snr = [1]
#a_snr = [1]
#a_snr = [1,2,5,10]
a_color = ['b','g','r','o']
#for snr in [1,10]:
fig = plt.figure()
ax = plt.subplot(111)
iter = 1
a_wf = []
maxpowerenv = 0
pnoise = 0
for snr,col in zip(a_snr,a_color):
    a_fitsnr = np.array([])
    for i in range(iter):
        sim = simulation.Simulation(snr =snr, siglength=signallength, det=det,sampling=5e9)
        sim.producetime()
        sim.producenoise(True)
        sim.produceconstantsignal()
        noisewf = waveform.Waveform(sim.time,sim.noise, type='hf')
        simwf = waveform.Waveform(sim.time,sim.noise+sim.signal, type='hf')
        noisewfsq = waveform.Waveform(sim.time,noisewf.amp**2/50, type='hf')
        simwfsq = waveform.Waveform(sim.time,simwf.amp**2/50, type='hf')
        wf = det.producesimwaveform(simwf,'adc',method)
        noisewf = det.producesimwaveform(noisewf,'adc',method)
        analyser = analyse.Analyse(det)
        powerwf = analyser.producepowerwaveform(wf)
        noisepowerwf = analyser.producepowerwaveform(noisewf)
        mean = analyser.getmean(simwfsq)
        meannoise = analyser.getmean(noisewfsq)
        mean2 = analyser.getmean(powerwf)
        meannoise2 = analyser.getmean(noisepowerwf)
        print mean/meannoise -1, ' mean2 - ', mean2/meannoise2 -1

# mean_wf = np.zeros(len(a_wf[1].amp))
# for w in a_wf:
#     mean_wf += w.amp
# #    plt.plot(w.amp,'b',alpha=0.2)
# mean_wf = mean_wf/len(a_wf)
# plt.plot(mean_wf,'r-',lw=1)
# print 'max mean' , np.max(mean_wf)
# print 'max power env = ', maxpowerenv

# x = a_wf[0].time/1e-6
# popt, pcov = curve_fit(func, x, mean_wf[:-1], p0=[np.max(mean_wf),0.03,5,mean])
# fitmean = func(x,popt[0],popt[1],popt[2],popt[3])
# plt.plot(fitmean,'g',lw=2)
# print popt
# print np.max(fitmean) - np.mean(fitmean[len(fitmean)/2:])
# #print np.max(mean_wf)/np.mean(mean_wf[len(mean_wf)/2:])
# #plt.legend()
plt.show()
