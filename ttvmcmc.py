import sys
import gzip
import acor
import multiprocessing as multi
from numpy import *
from fitnessNEW import *
import emcee
from emcee import PTSampler
import matplotlib.pyplot as pl
from argparse import ArgumentParser
#---------------------------------------------------
# Chain file handling
#--------------------------------------------------
def write_headers(Temps):
	for i,temp in enumerate(Temps):
		header = "# Chain {0:02d} of {1:02d}, temperature: {2:06.3f}\n".format(i,len(Temps),temp)
		with gzip.open('chain.{0:02d}.dat.gz'.format(i), 'w') as out:
			out.write(header)
            	with gzip.open('chain.{0:02d}.lnlike.dat.gz'.format(i), 'w') as out:
                	out.write(header)
            	with gzip.open('chain.{0:02d}.lnpost.dat.gz'.format(i), 'w') as out:
                	out.write(header)

def reset_files(NTs):
	for i in range(NTs):
            with gzip.open('chain.{0:02d}.dat.gz'.format(i), 'r') as inp:
                  header = inp.readline()
            with gzip.open('chain.{0:02d}.dat.gz'.format(i), 'w') as out:
                  out.write(header)
            
            with gzip.open('chain.{0:02d}.lnlike.dat.gz'.format(i), 'r') as inp:
                  header = inp.readline()
            with gzip.open('chain.{0:02d}.lnlike.dat.gz'.format(i), 'w') as out:
                  out.write(header)
            
            with gzip.open('chain.{0:02d}.lnpost.dat.gz'.format(i), 'r') as inp:
                  header = inp.readline()
            with gzip.open('chain.{0:02d}.lnpost.dat.gz'.format(i), 'w') as out:
                  out.write(header)
#------------------------------------------
#  Walker initilization
#------------------------------------------
#e,e1 = 0.07,0.05
#w,w1 = -1.0,2.0
p0 = zeros(4)
def initialize_walkers(nwalk,p0):
	return random.normal(size=(nwalk,4)) * 0.1 + p0
def logp(pars):
	if pars[0]**2 + pars[1]**2 >= 1 or pars[2]**2 + pars[3]**2 >= 1:
		return -inf
	return 0.0
#------------------------------------------
#  MAIN
#------------------------------------------
if __name__=="__main__":
#---------------------------------------------------------------------------------
#
# Will's method of repeatedly running and recentering until satisfied
#
# See: https://github.com/farr/nu-ligo-utils/tree/master/ensemble-sampler : run.py
#
#----------------------------------------------------------------------------------
	input_data=loadtxt("./inner.ttv")
	input_data1 = loadtxt("./outer.ttv")

	parser = ArgumentParser(description='run an ensemble MCMC analysis of a pair of TTVs')
	parser.add_argument('--restart', default=False, action='store_true', help='continue a previously-existing run')
	parser.add_argument('-n','--nensembles', metavar='N', type=int, default=100, help='number of ensembles to accumulate')
	parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of walkers to use')
	parser.add_argument('--ntemps', metavar='N', type=int, default=8, help='number different temperature scales to use')
	parser.add_argument('--nthin', metavar='N', type=int, default=10, help='number of setps to take between each saved ensemble state')
	parser.add_argument('--nthreads', metavar='N', type=int, default=multi.cpu_count(), help='number of concurrent threads to use')
	parser.add_argument('-f','--first_order', default=False, action='store_true', help='only compute first-order TTV contribution')

	args = parser.parse_args()
	restart = args.restart
	nensembles=args.nensembles
	nwalkers = args.nwalkers
	ntemps=args.ntemps
	nthin=args.nthin
	nthreads=args.nthreads
	firstFlag = args.first_order
	ndim=4
	
	# get fitness object
	ft = fitness(input_data,input_data1)
	def fit(x):
		return ft.fitness(x,firstOrder=firstFlag)[0]

	means=[]
	p = zeros((ntemps, nwalkers, ndim))
	if restart:
		for i in range(ntemps):
			p[i, :, :] = loadtxt('chain.%02d.dat.gz'%i)[-nwalkers:,:]

		means = list(mean(loadtxt('chain.00.dat.gz').reshape((-1, nwalkers, ndim)), axis=1))
	else:
		# Initialize chains
		p = [initialize_walkers(nwalkers,p0)]
      		for temp in range(ntemps-1):
            		p.append(initialize_walkers(nwalkers,p0))

	# initialize sampler
	sampler = emcee.PTSampler(ntemps,nwalkers,ndim,fit,logp,threads=nthreads)
	Ts = 1.0/sampler.betas
	if not restart:
		write_headers(Ts)
	print "Beggining ensemble evolution"
	print
	sys.stdout.flush()

	lnpost = None
	lnlike = None
	old_best_lnlike = None
	reset = False

#------------------------------------------------
# MAIN LOOP
#------------------------------------------------

	while True:
		for p,lnpost,lnlike in sampler.sample(p,lnprob0=lnpost, lnlike0=lnlike, iterations=nthin, storechain = False):
			pass	

		print 'acceptance fraction = ', ' '.join( map( lambda x: '{0:6.3f}'.format(x), mean(sampler.acceptance_fraction,axis = 1))) 
		print 'Tswap fraction      = ', ' '.join(map(lambda x: '{0:6.3f}'.format(x), sampler.tswap_acceptance_fraction))
		sys.stdout.flush()
		maxlnlike = max(lnlike[0])
		
		if old_best_lnlike is None:
			old_best_lnlike = maxlnlike
		
			if not restart:
                        # If on first iteration, then start centered around
                        # the best point so far
                        	imax = argmax(lnlike[0])
                        	best = p.reshape((-1, p.shape[-1]))[imax,:]
                        	p = recenter_best(p, best, logp, shrinkfactor=4.0,nthreads=nthreads)
                        
                        	lnpost = None
                        	lnlike = None
                        	sampler.reset()
                        
                        	continue

		if maxlnlike >  old_best_lnlike + p.shape[-1]/2.0:
			old_best_lnlike =  maxlnlike
			reset = True
			means = []

			imax = argmax(lnlike)
			best = p.reshape((-1,p.shape[-1]))[imax,:]
			p = recenter_best(p,best,logp,shrinkfactor=4.0,nthreads=nthreads)
			
			lnpost = None
			lnlike = None
			sampler.reset()

			print 'Found new best likelihood of {0:.1f}'.format(old_best_lnlike)
			print 'Resetting around parameters'
			print
			sys.stdout.flush()
			
			continue
		# Overwrite chains if resetting
		if reset:
			reset_files(ntemps)
			reset = False
		# Append current state to chain files
		for i in range(ntemps):
			with gzip.open('chain.{0:02d}.dat.gz'.format(i), 'a') as out:
                        	savetxt(out, p[i,:,:])
                	with gzip.open('chain.{0:02d}.lnlike.dat.gz'.format(i), 'a') as out:
                        	savetxt(out, lnlike[i,:].reshape((1,-1)))
                  	with gzip.open('chain.{0:02d}.lnpost.dat.gz'.format(i), 'a') as out:
                        	savetxt(out, lnpost[i,:].reshape((1,-1)))
		# Take the means over all walkers for zero temperature chain
		means.append(mean(p[0,:,:], axis=0))
		ameans = array(means)
		# toss out ~first 20% as burn-in
		ameans = ameans[int(round(0.2*ameans.shape[0])):, :]
		taumax = float('-inf')
            	for j in range(ameans.shape[1]):
                	try:
                		tau = acor.acor(ameans[:,j])[0]
                	except:
                        	tau = float('inf')
                  
                  	taumax = max(tau, taumax)
            
            	ndone = int(round(ameans.shape[0]/taumax))
            
            	print 'Computed {0:d} effective ensembles (max correlation length is {1:g})'.format(ndone, taumax)
            	print
            	sys.stdout.flush()
            
            	if ndone > nensembles:
            		break
