########################################
## produce a noise AND SIGNAL waveform #
## according  the detector type and the#
## method  given in argument          ##
## (defaut is norsat) method 3        ##
## The signal has a gaussian enveloppe##
########################################
from mod import *
from scipy.optimize import curve_fit
#def func(x, a, sigma,mu,b):
#    return (a/sigma)*np.exp(-((x - mu)/sigma)**2) + b
#    return a*np.exp(-((x - mu)/sigma)**2) + b

def func(x, a, sigma,mu,b):
    return np.log10((a/sigma)*np.exp(-((x - mu)/sigma)**2)+1e-10) + b
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
temp = 10
#a_snr = [1,2,5,10]
a_snr = [0.001]
a_color = ['b','g','r','o']
#for snr in [1,10]:
fig = plt.figure()
ax = plt.subplot(111)
iter = 10
snr = 0.001
meanmeanadc = np.array([])
a_temp = np.array([])
#for temp in [1,10,100]:
det = detector.Detector(temp=temp*10,type=dettype)
det.loadspectrum()
meanadc = np.array([])

for i in range(iter):
    sim = simulation.Simulation(snr =snr, siglength=signallength, det=det,sampling=5e9)
    sim.producetime()
    sim.producenoise(True)
    sim.setpowerenvelope('gauss')
    sim.producesignal()        
    simwf = waveform.Waveform(sim.time,sim.noise+sim.signal, type='hf')
    power = simwf.amp**2
    

    wf = det.producesimwaveform(simwf,'adc',method)
    analyser = analyse.Analyse(det)
    powerwf = analyser.producepowerwaveform(wf)
    meanadc = np.append(meanadc,np.mean(powerwf.amp))
    #        meanadc = np.append(meanadc,np.mean(wf.amp))
    print meanadc
    meanmeanadc = np.append(meanmeanadc,np.mean(meanadc))
    a_temp = np.append(a_temp,det.pnoise*det.gain)

plt.plot(10*np.log10(a_temp),10*np.log10(meanmeanadc),'o')
#plt.plot(10*np.log10(a_temp),meanmeanadc)
#fit = np.polyfit(10*np.log10(a_temp),meanmeanadc,1)
fit = np.polyfit(10*np.log10(a_temp),10*np.log10(meanmeanadc),1)
print fit
#ax.set_xlabel('time [us]')
#ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
#ax.set_ylabel('power [W]')
#plt.legend()
plt.show()
