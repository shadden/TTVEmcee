TTVFAST_PATH = "/projects/b1002/shadden/7_AnalyticTTV/03_TTVFast/PyTTVFast"
#TTVFAST_PATH = "/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface"

import sys
sys.path.insert(0, '/Users/samuelhadden/13_HighOrderTTV/TTVEmcee')

import gzip
import acor
import multiprocessing as multi
from numpy import *
from fitnessNEW import *
import emcee
from emcee import PTSampler
import matplotlib.pyplot as pl
from argparse import ArgumentParser

#------------------------------------------
#  MAIN
#------------------------------------------
if __name__=="__main__":

	parser = ArgumentParser(description='run an ensemble MCMC analysis of a pair of TTVs')
	parser.add_argument('--restart', default=False, action='store_true', help='continue a previously-existing run')
	parser.add_argument('--erase', default=False, action='store_true', help='Start walkers from old files but overwrite them')
	parser.add_argument('-n','--nensembles', metavar='N', type=int, default=100, help='number of ensembles to accumulate')
	parser.add_argument('--nacor', metavar='N', type=int, default=5, help='Number of consecutive non-infinite autocorellation measurements required before exiting')
	parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of walkers to use')
	parser.add_argument('--nthin', metavar='N', type=int, default=10, help='number of setps to take between each saved ensemble state')
	parser.add_argument('--nthreads', metavar='N', type=int, default=multi.cpu_count(), help='number of concurrent threads to use')
	parser.add_argument('-P','--parfile', metavar='FILE', default=None, help='Text file containing parameter values to initialize walker around.')
	parser.add_argument('--noloop', default=False, action='store_true', help='Run set-up but do not excecute the MCMC main loop')
		
	input_data = loadtxt("./inner.ttv")
	input_data1= loadtxt("./outer.ttv")
	while min(append(input_data[:,0],input_data1[:,0])) != 0:
		print "re-numbering transits..."
		input_data[:,0] -= 1
		input_data1[:,0] -= 1

#----------------------------------------------------------------------------------
# Get input TTV data and remove outliers
#----------------------------------------------------------------------------------
	input_dataTR,input_data1TR = TrimData(input_data,input_data1,tol=2.5)
	if len(input_dataTR) != len(input_data):
		print "Removed %d transit(s) from inner planet:" %( len(input_data) - len(input_dataTR) )
		for bad in set(input_data[:,0]).difference( set(input_dataTR[:,0]) ):
			print "\t%d"%bad
		input_data = input_dataTR
	if len(input_data1TR) != len(input_data1):
		print "Removed %d transits from outer planet:" %( len(input_data1) - len(input_data1TR) )
		for bad in set(input_data1[:,0]).difference( set(input_data1TR[:,0]) ):
			print "\t%d"%bad
		input_data1 = input_data1TR
#----------------------------------------------------------------------------------

	args = parser.parse_args()
	restart = args.restart
	nensembles=args.nensembles
	nwalkers = args.nwalkers
	nthin=args.nthin
	nthreads=args.nthreads

#----------------------------------------------------------------------------------
	if args.parfile != None:
		try:
			pars0 = loadtxt(args.parfile)
		except IOError:
			print "Parameter file %s not found!"%args.parfile
			print "Aborting..."
			sys.exit()
	else:
		pars0 = array([1.e-5,1.e-5,0,0,0,0])
#----------------------------------------------------------------------------------
	sys.path.insert(0,TTVFAST_PATH)
	import PyTTVFast as ttv
	ndim = 2*5-2
	nbody_fit = ttv.TTVFitnessAdvanced([input_data,input_data1])
#----------------------------------------------------------------------		
	# Priors function
	def logp(pars):
		# Masses must be positive
		masses = array((pars[0],pars[3]))
		bad_masses = any(masses < 0.0)
		if bad_masses:
			return -inf
		
		# Mean anomalies lie between -pi and pi
		bad_angles = abs(pars[-1]) > pi
		if bad_angles:
			return -inf
		
		# Eccentricities must be smaller than 1
		exs,eys = array([pars[1],pars[4]]),array([pars[2],pars[5]])
		bad_eccs = any(exs**2 +eys**2 >= 1.0)
		if bad_eccs:
			return -inf

		return 0.0

	def fit(x):
		return nbody_fit.CoplanarParametersFitness(x) + logp(x)

	def initialize_walkers(nwalk,p0):
		"""Initialize walkers around point p0 = [ mass1,mass2,ex1,ey1,ex2,ey2,P2_obs/P1_obs, dL_obs]"""
		ics = nbody_fit.GenerateRandomInitialConditions([p0[0],p0[1]],0.1,[[p0[2],p0[3]],[p0[4],p0[5]]],0.005,nwalk)
		return array([ nbody_fit.convert_params(ic) for ic in ics])

	means=[]
	p = zeros((nwalkers, ndim))
#-----------------------------------------------------------------
#	Set up sampler and walkers
#-----------------------------------------------------------------	
	if restart:
		# Read in old walkers
		print "Loading chain from file..."
		p = loadtxt('chain.dat.gz')[-nwalkers:,:]
		print "%d x %d chain loaded"%p.shape
	else:
		# Initialize new walkers
		p=initialize_walkers(nwalkers,pars0)
	
	for x in p.reshape(-1,8):
		assert logp(x)==0.0 and fit(x) > -inf, "Bad IC generated!"
		

	# initialize sampler
	sampler = emcee.EnsembleSampler(nwalkers,ndim,fit,threads=nthreads)

	print "Beggining ensemble evolution"
	print "Running with %d parallel threads" % nthreads
	print

	sys.stdout.flush()
	lnpost = None
	lnlike = None
	old_best_lnlike = None
	reset = False
	Nmeasured = 0
	if not restart or args.erase:
		with gzip.open('chain.dat.gz', 'w') as out:
				out.write("# Parameter Chains")
		with gzip.open('chain.lnlike.dat.gz', 'w') as out:
				out.write("# Likelihoods")

#------------------------------------------------
# MAIN LOOP
#------------------------------------------------

	for k in range( int(ceil(nensembles/nthin)) ):
		if args.noloop:
			break
		# take 'nthin' samples.
		for p,lnlike,blobs in sampler.sample(p,iterations=nthin, storechain = True):
			pass	

		print 'acceptance fraction = %.3f'%mean(sampler.acceptance_fraction) 
		print 'Autocorrelation lengths: '
		print 'M1\t EX1\t EY1\t M2\t EX2\t EY2\t P\t L'
		print '\t'.join(map(lambda x: '{0:.1f}'.format(x), sampler.acor))
		sys.stdout.flush()

		maxlnlike = max(lnlike)
		
		if old_best_lnlike is None:
			old_best_lnlike = maxlnlike
			print 'Found best likelihood of {0:.1f}'.format(old_best_lnlike)

		if maxlnlike >  old_best_lnlike + p.shape[-1]/2.0:
			old_best_lnlike =  maxlnlike
			print 'Found new best likelihood of {0:.1f}'.format(old_best_lnlike)
			print
			sys.stdout.flush()
			
			continue

		# Append current state to chain file
		with gzip.open('chain.dat.gz', 'a') as out:
			savetxt(out, p)
		with gzip.open('chain.lnlike.dat.gz', 'a') as out:
			savetxt(out, lnlike)
