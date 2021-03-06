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
parser.add_argument("--outfolder", type=str, nargs='?', default=constant.outfolder, help="pkl out folder")

args = parser.parse_args()
print '#####################################'
print '###### detector: ', args.det ,' ######'
print '###### power det method: ', args.method ,' ######'
print '###### system temp: ', args.tsys ,' #########'
print '###### data folder ', args.folder ,' #########'
print '###### out folder ', args.outfolder ,' #########'
print '#####################################'

dettype = args.det
method = args.method
tsys = args.tsys
databasefolder = args.folder
outfolder = args.outfolder

det = detector.Detector(type=dettype,temp=tsys)
det.loadspectrum()

#yields = [200]
#yields = [1]
yields = [1,10,100,200]
if dettype == 'norsat':
    datafolder = databasefolder + '/cband/'
elif dettype == 'helix':
    datafolder = databasefolder + '/lband/'
names = glob.glob(datafolder + '*')
iter = 1


for y in yields:
    for i in range(iter):
        a_res = []
        if dettype == 'norsat':
            outname = outfolder + '/cband/' + 'out_yieldfact_' + str(y) + '.pkl'
        elif dettype == 'helix':      
            outname = outfolder + '/lband/' + 'out_yieldfact_' + str(y) + '.pkl'
        for n in names:
#        for n in names[:1000]:
#        for n in names[:2]:
            ev = event.Event(fname=n, type='test')
            ev.loadevent()
            for ant in ev.antennas:
                time = ant.maketimearray()
                power = ant.power
                sim = simulation.Simulation(det=det,sampling=5e9)
                sim.producetime()
                sim.producenoise(True)
                sim.setpowerenvelopewitharray([time,y*power])
                if np.max(sim.powerenvelope) < 1e-14:
                    continue
                
#                print 'mean envelope',  np.mean(sim.powerenvelope)
#                if np.mean(sim.powerenvelope) == 0.0:
#                    print 'skipping'
#                    continue
                sim.producesignal()
                simwf = waveform.Waveform(sim.time,sim.noise+sim.signal, type='hf')
        
                wf = det.producesimwaveform(simwf,'adc',method)
                analyser = analyse.Analyse(det)
                powerwf = analyser.producepowerwaveform(wf)
#                print np.mean(powerwf.amp)
                meanwf = analyser.producemeanwaveform(powerwf)
                sigmawf = analyser.producesigmawaveform(powerwf)
                envelopewf = waveform.Waveform(sim.time,sim.powerenvelope)
                cc = analyser.crosscorrel(meanwf,envelopewf)
                cc = analyser.producesigmawaveform(cc)        
#                ant.enveloppe = envelopewf
#                ant.adcwf_power = powerwf
                ant.adcwf_sigma = sigmawf
                ant.maxenvelope = np.max(sim.powerenvelope)
#                ant.filterwf_sigma = cc
                res = results.Results()
                res.setevid(ev.id)
                res.shower= ev.shower
                res.ant = ant
                res.y = y
                a_res.append(res)
#                plt.plot(envelopewf.amp)
                
#                plt.plot(sigmawf.amp)
    output = open(outname, 'wb')
    pickle.dump(a_res,output)
    output.close()

plt.show()
