from argparse import ArgumentParser
import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import curve_fit

parser = ArgumentParser(description='Analyze the auto-correlation lengths of an MCMC run')
#
parser.add_argument('--nwalkers','-N',type=int,default=300,\
	help='Number of walkers used, needed to resconstruct autocorrelation information')
parser.add_argument('--dryrun','-D',default=False,action='store_true',\
	help='Do not load chain files or compute any information.')
parser.add_argument('--maxlag','-L',type=int,default=30,\
	help='Maximum lag value to use in computing autocorrelation lengths')

args = parser.parse_args()
nwalkers=args.nwalkers

def acorfn(a,lag):
	mu = np.mean(a)
	if lag==0:
		cff = np.mean(a*a) - mu**2
	else:
		cff = np.mean(a[:-lag] * a[lag:]) - mu**2
	return cff

def expfn(data,tau):
	return np.exp(-data / tau)

def parameterAcor(chain,ipar,maxlag=args.maxlag,plot_data=False):
	all_data = []
	for walker in range(nwalkers):
		pchain = chain[:,walker,ipar]
		norm = acorfn(pchain,0)
		data = np.array([(lag,acorfn(pchain,lag)/norm) for lag in range(maxlag)])
		all_data.append(data)
	
	all_data = np.array(all_data)
	mean_data = np.mean(all_data,axis=0)
	tau = curve_fit(expfn,mean_data[:,0],mean_data[:,1])[0][0]
	if plot_data:
		base_line,=pl.plot(mean_data[:,0],mean_data[:,1])
		pl.plot(mean_data[:,0],expfn(mean_data[:,0],tau),'--',c=base_line.get_color())
	
	return tau

def shape_chain(chain,nwalk,pct_drop=0.1):
	npars = chain.shape[-1]
	npl = (npars+2) / 5
	chainlength=chain.shape[0]
	ntake = int( np.ceil( (1-pct_drop) * chainlength/nwalk ) * nwalk )
	schain = chain[-ntake:].reshape(-1,nwalk,npars)
	return schain
if not args.dryrun:
	lnlike,chain = np.loadtxt('./chain.lnlike.dat.gz'),np.loadtxt('./chain.dat.gz')
	schain = shape_chain(chain,nwalkers,0.1)
	
	nplanets = (chain.shape[-1]+2) / 5
	taus = []
	for ipar in range(3*nplanets):
		taus.append( parameterAcor(schain,ipar,plot_data=True) )

	
	with open("correlation_lengths.txt","w") as fi:
		fi.write("N: %d \t Nwalk: %d\n"%(schain.shape[0],schain.shape[1]))
		fi.write( "param \t tau \t N_eff \n")
		for i,tau in enumerate(taus):
			fi.write( "%d \t %.1f \t %.1f \n"%(i,tau, nwalkers*schain.shape[0]/tau) )
			print i, tau, schain.shape[0]/tau
	
	pl.show()
	
