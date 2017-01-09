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

temp = 100
det = detector.Detector(temp=temp,type=dettype)
det.loadspectrum()
a_snr = [5]
a_siglength = [100e-9]
#a_snr = [1,2,5,10]
#a_siglength = [10e-9, 50e-9, 100e-9, 500e-9]
a_color = ['b','g','r','o']
iter = 1
a_wf = []
maxpowerenv = 0
pnoise = 0
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)
for siglength,col in zip(a_siglength,a_color):
    a_meassnr = np.array([])
    a_measerrsnr = np.array([])
    for snr in a_snr:
        a_fitsnr = np.array([])
        a_fitsnr2 = np.array([])
        a_errfitsnr2 = np.array([])
        for i in range(iter):
            sim = simulation.Simulation(snr =snr, siglength=siglength, det=det,sampling=5e9)
            sim.producetime()
            sim.producenoise(True)
            sim.setpowerenvelope('gauss')
            maxpowerenv =  np.max(sim.powerenvelope)
            sim.producesignal()
            simwf = waveform.Waveform(sim.time,sim.noise+sim.signal, type='hf')
            simwfsq = waveform.Waveform(sim.time,simwf.amp**2/50, type='hf')
            wf = det.producesimwaveform(simwf,'adc',method)
#        wf = det.producesimwaveform(simwf,'board',method)
            analyser = analyse.Analyse(det)
            powerwf = analyser.producepowerwaveform(wf)
            mean = analyser.getmean(simwfsq)
            mean2 = analyser.getmean(powerwf)
            x = simwfsq.time/1e-6
            x2 = powerwf.time/1e-6
            popt, pcov = curve_fit(func,x , simwfsq.amp, p0=[snr*mean,0.03,5,mean])
            popt2, pcov2 = curve_fit(func,x2 , powerwf.amp, p0=[snr*mean2,0.03,5,mean2])
            fit = func(x,popt[0],popt[1],popt[2],popt[3])
            fit2 = func(x2,popt2[0],popt2[1],popt2[2],popt2[3])
            meassnr = (analyser.getmax(powerwf) -mean)/mean
            fitsnr = (popt[0])/popt[3]
            fitsnr2 = (popt2[0])/popt2[3]
#            print 'fitsnr = ', fitsnr, ' fitsnr2 = ', fitsnr2 
            a_wf.append(powerwf)
            a_fitsnr = np.append(a_fitsnr,(fitsnr - snr)/snr)
            a_fitsnr2 = np.append(a_fitsnr2,(fitsnr2 - snr)/snr)
            pnoise = det.pnoise
        a_meassnr  = np.append(a_meassnr,np.mean(a_fitsnr2))
        a_measerrsnr  = np.append(a_measerrsnr,np.std(a_fitsnr2)/np.sqrt(iter))
    ax.errorbar(a_snr,a_meassnr,yerr=a_measerrsnr,fmt='-o',label=str(siglength/1e-9)+' ns' )
ax.set_xscale('log')
ax.set_xlabel('input SNR')
ax.set_ylabel('(output SNR - input SNR)/input SNR ')
plt.legend()
fig2 = plt.figure()
ax1 = plt.subplot(111)
plt.plot(x2,powerwf.amp/mean2 -1 ,lw=2,label='power trace')
plt.plot(x2,fit2/mean2 -1,lw=2,label='fit')
plt.plot(x, ( sim.powerenvelope+ pnoise )/ pnoise -1,lw=2,label='input power envelope')
plt.xlabel('time [us]')
plt.ylabel('power [SNR]')
plt.legend()
plt.xlim(3,8)
# print 'power ', np.mean(a_fitsnr) , ' +- ' , np.std(a_fitsnr)
# print  'sim power', np.mean(a_fitsnr2) , ' +- ' , np.std(a_fitsnr2)
# plt.hist(a_fitsnr,histtype='step',lw=2)
# plt.hist(a_fitsnr2,histtype='step',lw=2)
plt.show()
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
