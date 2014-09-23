import sys
sys.path.insert(0, '/Users/samuelhadden/13_HighOrderTTV/TTVEmcee')
sys.path.insert(0, '/projects/b1002/shadden/7_AnalyticTTV/01_MCMC/00_source_code')
import fitnessNEW as fff
import triangle
from argparse import ArgumentParser
parser =  ArgumentParser(description='Postprocessing for mcmc runs') 
parser.add_argument('--plot', default=False, action='store_true', help='Run plotting routines') 
parser.add_argument('--parsplot', default=False, action='store_true', help='plot paramater corner') 
parser.add_argument('--zplot', default=False, action='store_true', help='plot Z corner') 
parser.add_argument('--chainplot', default=False, action='store_true', help='plot walker chains') 
parser.add_argument('--bestplot', default=False, action='store_true', help='plot TTVs from best-fit and true parameters') 
args = parser.parse_args()

parsplot = args.parsplot
zplot = args.zplot
chainplot = args.chainplot
bestplot = args.bestplot
if args.plot:
	parsplot = True
	zplot = True
	chainplot = True
	bestplot = True


nwalkers = 200
replot = False
data = loadtxt('chain.00.dat.gz')
lnlike = loadtxt('chain.00.lnlike.dat.gz').reshape(-1,)

p,t0,p1,t10 = fff.fullPeriodFit2(loadtxt('inner.ttv'),loadtxt('outer.ttv'))
j=fff.get_res(p1/p)
delta = (j-1.) * p1 / (j*p) -1.
if j==3:
	fCoeff = -2.02522
	f1Coeff = 2.48401
elif j==2:
	fCoeff = -1.19049
	f1Coeff = 0.42839 
m,m1,ex,ey,ex1,ey1 = data.T

bestpars = data[argmax(lnlike.reshape((-1,)))]
truepars = loadtxt("TrueParams.txt")
input_data=loadtxt("./inner.ttv")
input_data1 = loadtxt("./outer.ttv")
	
ft = fff.fitness(input_data,input_data1)
j = ft.j
delta = ft.pratio *(j-1.)/j - 1.
fCoeff = ft.f[j-1]
f1Coeff = ft.f1Int[j-1]	
def Zfn(exx,eyy,exx1,eyy1):
	Zxx = fCoeff * exx + f1Coeff * exx1
	Zyy = fCoeff * eyy + f1Coeff * eyy1
	#
	Vxx = fCoeff  +   1.5 * Zxx / (delta)
	Vyy =  1.5 * Zyy / (delta)
	#
	return (Zxx,Zyy,sqrt(Zxx**2+Zyy**2),arctan2(Vyy,Vxx) )


Zx,Zy,absZ,argV = Zfn(ex,ey,ex1,ey1)
Zxtrue,Zytrue,absZtrue,argVtrue=Zfn(truepars[2],truepars[3],truepars[4],truepars[5])

data2 = array([log10(data[:,0]),log10(data[:,1]),data[:,2],data[:,3],data[:,4],data[:,5]]).T
data3 = array([(data[:,0]),(data[:,1]),Zx,Zy,arctan2(Zy,Zx),absZ]).T

true_vals = truepars.copy()
true_vals[:2] = log10(true_vals[:2])

if parsplot:
	#elims = (-1,1)
	elims = (-3*absZtrue,3*absZtrue)
	extnts = [1.,1.,elims,elims,elims,elims]
	lbls = ['logM','logM1','ex','ey','ex1','ey1']
	triangle.corner(data2,extents=extnts,labels=lbls,truths=true_vals)
	show()
	savefig('parameters_corner.png')
if zplot:	
	lbls3 = ['logM','logM1','Zx','Zy','arg Z','|Z|']
	truths3= array([truepars[0],truepars[1],Zxtrue,Zytrue,arctan2(Zytrue,Zxtrue),absZtrue])
	zxtnts = (-3*absZtrue,3*absZtrue)
	#triangle.corner(data3,labels=lbls3,truths=truths3,extents=[(1.e-6,1.e-4),(1.e-6,1.e-4),zxtnts,zxtnts,(-pi,pi),(0,3*absZtrue)])
	triangle.corner(data3,labels=lbls3,truths=truths3)
	show()
	savefig('mass-vs-Z_corner.png')
if chainplot:	
	figure()
	panels = []
	for fignum in range(6):
		panels.append( subplot(321+fignum) )
		ylabel(lbls[fignum])
	
	parameter_chains = data2.reshape(-1,nwalkers,6)
	lnlike_chains = lnlike.reshape(-1,nwalkers).T
	for walkerN in range(nwalkers):
		#panel1.plot( lnlike_chains[walkerN])
		for fignum in range(6):
			panels[fignum].plot( parameter_chains[:,walkerN,fignum] )
	show()
	savefig('chains_plot.png')
if bestplot:	
	ft.fitplot([truepars,bestpars])
	subplot(211)
	title(" lnlike: %.2f (%.2f)" % (ft.fitness2(bestpars),ft.fitness2(truepars)) )
	savefig('best_vs_true.png')
	
	
	ft.residfitplot([truepars,bestpars])
	subplot(211)
	title(" lnlike: %.2f (%.2f)" % (ft.fitness2(bestpars),ft.fitness2(truepars)) )
	savefig('best_vs_true_resids.png')

ecc = sqrt(ex**2+ey**2)
eccTrue = sqrt(truepars[2]**2+truepars[3]**2)
eccBest = sqrt(bestpars[2]**2+bestpars[3]**2)
ecc1 = sqrt(ex1**2+ey1**2)
ecc1True = sqrt(truepars[4]**2+truepars[5]**2)
ecc1Best = sqrt(bestpars[4]**2+bestpars[5]**2)
with open("summary.txt",'w') as fi:
	fi.write('\t|True|\t Best|\t median|\t25%|\t75%|\n')
	fi.write("m\t|%.2f\t|%.2f\t|%.2f\t|%.2f\t|%.2f\n"%( log10(truepars[0]), log10(bestpars[0]),median(log10(m)),percentile(log10(m),25),percentile(log10(m),75) ) )
	fi.write("m1\t|%.2f\t|%.2f\t|%.2f\t|%.2f\t|%.2f\n"%( log10(truepars[1]), log10(bestpars[1]),median(log10(m1)),percentile(log10(m1),25),percentile(log10(m1),75) ) )
	fi.write("e\t|%.3f\t|%.3f\t|%.3f\t|%.3f\t|%.3f\n"%(eccTrue ,eccBest ,median(ecc),percentile(ecc,25),percentile(ecc,75) ) )
	fi.write("e1\t|%.3f\t|%.3f\t|%.3f\t|%.3f\t|%.3f\n"%(ecc1True ,ecc1Best ,median(ecc1),percentile(ecc1,25),percentile(ecc1,75) ) )
	fi.write("Z\t|%.3f\t|%.3f\t|%.3f\t|%.3f\t|%.3f\n"%( Zfn(truepars[2],truepars[3],truepars[4],truepars[5])[-2] ,Zfn(bestpars[2],bestpars[3],bestpars[4],bestpars[5])[-2] ,median(absZ),percentile(absZ,25),percentile(absZ,75) ) )
	fi.write("Delta: %.3f\n"%delta)
