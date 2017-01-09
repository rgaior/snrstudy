from mod import *

parser = argparse.ArgumentParser()
parser.add_argument("--det", type=str, nargs='?',default='norsat', help="type of detector: gi, dmx, norsat, helix")
parser.add_argument("--tsys", type=float, nargs='?', default=50, help="system noise temperture (integer)")
parser.add_argument("--folder", type=str, nargs='?', default=constant.outfolder, help="folder where the pkl array are located")

args = parser.parse_args()
print '#####################################'
print '###### detector: ', args.det ,' ######'
print '###### system temp: ', args.tsys ,' #########'
print '###### data folder ', args.folder ,' #########'
print '#####################################'

dettype = args.det
tsys = args.tsys
folder = args.folder
plotrep = '/Users/romain/Dropbox/Public/EASIER/gigas/plots/20160708/'

yields = [100]
#yields = [1,10,100]
#yields = [1,10,100,200]
if dettype =='norsat':
    titlename = 'C band'
    plotname = 'cband' 
elif dettype =='helix':
    titlename = 'L band'
    plotname = 'lband' 
histfig = plt.figure()
ax0 = plt.subplot(111)
maxvstfig = plt.figure()
ax1 = plt.subplot(111)
tracefig = plt.figure(figsize=(12,6))
ax2 = plt.subplot(111)
a_max = np.array([])
a_delt = np.array([])
evcounter = 0
evintimecounter = 0
for y in yields:
    if dettype =='norsat':
        fname = folder + '/cband/''out_yieldfact_' + str(y) + '.pkl'
    elif dettype =='helix':
        fname = folder + '/lband/''out_yieldfact_' + str(y) + '.pkl'
    print fname
    histfig.suptitle(titlename + ', ' + str(y)+ '*yield',fontsize=15, fontweight='bold')
    maxvstfig.suptitle(titlename + ', ' + str(y)+ '*yield',fontsize=15, fontweight='bold')
    tracefig.suptitle(titlename + ', ' + str(y)+ '*yield',fontsize=15, fontweight='bold')
    pkl_file = open(fname, 'rb')
    a_res = pickle.load(pkl_file)
    pkl_file.close()
    print len(a_res)
    for res in a_res:
#        plt.plot(res.ant.adcwf_sigma.amp)
        max = res.ant.adcwf_sigma.getmax()
#        max = res.ant.filterwf_sigma.getmax()
#        print res.ant.filterwf_sigma
        tofmax = res.ant.adcwf_sigma.gettimeofmax()
        th_tofmax = 5e-6
        if (np.isnan(max) == True):
            print 'a cheese nan here '
            continue
        a_max = np.append(a_max,max)
        a_delt = np.append(a_delt,tofmax - th_tofmax)
        if max > 9:
            ax2.plot(res.ant.adcwf_sigma.amp)
            evcounter +=1  
            if np.absolute(tofmax - th_tofmax) < 200e-9:
                evintimecounter +=1  
            #            print 'time of max = ', res.ant.adcwf_sigma.gettimeofmax(), ' and max = ', max
#            print res.shower.energy
#        print res.ant.adcwf_sigma.getmax()
 
    bins = np.arange(np.min(a_max)-1,np.max(a_max)+3,2)
    n, bins, patches = ax0.hist(a_max,bins=bins,lw=2,histtype='step',log=True) 
    ax1.plot(a_delt,a_max,'.')       
    namehistfig = 'hist_' + str(y) + 'yield'+ plotname+'.png'
    namemaxvstfig = 'maxvst_' + str(y) + 'yield'+ plotname+'.png'
    nametracefig = 'trace_' + str(y) + 'yield'+ plotname+'.png'
    histfig.savefig(plotrep+namehistfig)
    maxvstfig.savefig(plotrep+namemaxvstfig)
    tracefig.savefig(plotrep+nametracefig)

print 'event with max > 9 sigma = ' , evcounter
print 'event in time with max > 9 sigma = ' , evintimecounter
plt.show()
