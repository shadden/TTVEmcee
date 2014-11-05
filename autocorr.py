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

def parameterAcor(chain,ipar,maxlag=30,plot_data=False):
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

if not args.dryrun:
	lnlike,chain = np.loadtxt('./chain.lnlike.dat.gz'),np.loadtxt('./chain.dat.gz')
	npars = chain.shape[-1]
	nplanets = (npars + 2) / 5
	chainlength = chain.shape[0]
	ntake = int(np.ceil(0.9 * chainlength/nwalkers) * nwalkers)
	schain = chain[-ntake:].reshape(-1,nwalkers,npars)
	
	taus = []
	for ipar in range(3*nplanets):
		taus.append( parameterAcor(schain,ipar,plot_data=True) )

	
	for i,tau in enumerate(taus):
		print i,tau
	
	pl.show()