import os
who =os.popen("whoami") 
if who.readline().strip() =='samuelhadden':
	print "On laptop..."
	TTVFAST_PATH = "/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface"
	ANALYTIC_TTV_PATH = "/Users/samuelhadden/13_HighOrderTTV/TTVEmcee"
else:
	print "On Quest..."
	TTVFAST_PATH = "/projects/b1002/shadden/7_AnalyticTTV/03_TTVFast/PyTTVFast"
	ANALYTIC_TTV_PATH = "/projects/b1002/shadden/7_AnalyticTTV/01_MCMC/00_source_code"
who.close()

import sys
sys.path.insert(0, '/Users/samuelhadden/13_HighOrderTTV/TTVEmcee')

import gzip
import acor
import multiprocessing as multi
from numpy import *
from fitnessNEW import *
import emcee
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
	parser.add_argument('--nburn', metavar='N', type=int, default=300, help='number samples to use for burn-in')
	parser.add_argument('--nacor', metavar='N', type=int, default=5, help='Number of consecutive non-infinite autocorellation measurements required before exiting')
	parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of walkers to use')
	parser.add_argument('--nthin', metavar='N', type=int, default=10, help='number of setps to take between each saved ensemble state')
	parser.add_argument('--nthreads', metavar='N', type=int, default=multi.cpu_count(), help='number of concurrent threads to use')
	parser.add_argument('-P','--parfile', metavar='FILE', default=None, help='Text file containing parameter values to initialize walker around.')
	parser.add_argument('--noloop', default=False, action='store_true', help='Run set-up but do not excecute the MCMC main loop')
	parser.add_argument('--analytic',default=False,action='store_true', help='Run MCMC using analytic TTV formula to compute the likelihood of parameters')
	parser.add_argument('--first_order',default=False,action='store_true', help='Use only the 1S term for analytic TTVs')
	parser.add_argument('--input','-I',metavar='FILE',default='planets.txt',help='File that lists the names of the files containing input transits')

	#----------------------------------------------------------------------------------
	# command-line arguments:

	args = parser.parse_args()
	restart = args.restart
	nensembles=args.nensembles
	nwalkers = args.nwalkers
	nthin=args.nthin
	nthreads=args.nthreads
	nburn = args.nburn
	infile = args.input
	nbody = (not args.analytic)
	first_order = args.first_order

	#----------------------------------------------------------------------------------

	
	with open(infile,'r') as fi:
		infiles = [ line.strip() for line in fi.readlines()]
	
	input_data =[]
	for file in infiles:
		input_data.append( loadtxt(file) )
	nplanets = len(input_data)


	while min( array([ min(tr[:,0]) for tr in input_data])  ) != 0:
		print "re-numbering transits..."
		for data in input_data:
			data[:,0] -= 1

	#----------------------------------------------------------------------------------
	# Get input TTV data and remove outliers
	#----------------------------------------------------------------------------------
	trim = None
	if trim:
		input_dataTR = TrimData(input_data,tol=3.)
		for i,new_transits in enumerate(input_dataTR):
			if len(new_transits) != len(input_data[i]):
				print "Removed %d transit(s) from inner planet:" %( len(input_data[i]) - len(new_transits) )
	
				for bad in set(input_data[i][:,0]).difference( set(input_dataTR[i][:,0]) ):
					print "\t%d"%bad

	if args.parfile != None:
		try:
			pars0 = loadtxt(args.parfile)
		except IOError:
			print "Parameter file %s not found!"%args.parfile
			print "Aborting..."
			sys.exit()

	else:
		pars0 = append(ones(nplanets) * 6.e-6 , zeros(2*nplanets) )
#----------------------------------------------------------------------------------
	sys.path.insert(0,TTVFAST_PATH)
	import PyTTVFast as ttv
	ndim = 5*nplanets-2

	#----------------------------------------------------------------------		
	#	Compute log-likelihoods from N-body integration
	#----------------------------------------------------------------------
	if nbody:
		sys.path.insert(0,TTVFAST_PATH)
		import PyTTVFast as ttv
		nbody_fit = ttv.TTVFitnessAdvanced(input_data)

		# Likelihood function
		#--------------------------------------------
		def fit(x):
			# Masses must be positive
			masses = (x[::3])[:nplanets]
			bad_masses = any(masses < 0.0)
			if bad_masses:
				return -inf
	
			# Mean anomalies lie between -pi and pi
			meanAnoms = x[3*nplanets+1::2]
			bad_angles = any( abs(meanAnoms) > pi)
			if bad_angles:
				return -inf
	
			# Eccentricities must be smaller than 1
			exs,eys =(x[1::3])[:nplanets],(x[2::3])[:nplanets]
			bad_eccs = any(exs**2 +eys**2 >= 0.9**2)
			if bad_eccs:
				return -inf
	
			return nbody_fit.CoplanarParametersFitness(x)
	
		# Walker initialization
		#--------------------------------------------
		def initialize_walkers(nwalk,p0):
			"""Initialize walkers around point p0 = [ mass1,mass2,...,ex1,ey1,ex2,ey2,...]""" #...,P2_obs/P1_obs,..., dL2_obs,...]"""
			masses = p0[:nplanets]
			evecs = p0[nplanets:].reshape(-1,2)
			ics = nbody_fit.GenerateRandomInitialConditions(masses,0.1,evecs,0.005,nwalk)
			return array(ics)
				

	#----------------------------------------------------------------------		
	#	Compute log-likelihoods from analytic formula
	#----------------------------------------------------------------------
	else:
		sys.path.insert(0,ANALYTIC_TTV_PATH)
		import AnalyticMultiplanetFit as ttv
		analytic_fit = ttv.MultiplanetAnalyticTTVSystem(input_data)

		# Likelihood function
		#--------------------------------------------
		def fit(x):
			# Masses must be positive
			masses = (x[::3])[:nplanets]
			bad_masses = any(masses < 0.0)
			if bad_masses:
				return -inf
	
			# Mean anomalies lie between -pi and pi
			meanAnoms = x[3*nplanets+1::2]
			bad_angles = any( abs(meanAnoms) > 2*pi)
			if bad_angles:
				return -inf
	
			# Eccentricities must be smaller than 1
			exs,eys =(x[1::3])[:nplanets],(x[2::3])[:nplanets]
			bad_eccs = any(exs**2 +eys**2 >= 0.9**2)
			if bad_eccs:
				return -inf
	
			return analytic_fit.parameterFitness(x,Only_1S=first_order)

		# Walker initialization
		#--------------------------------------------
		def initialize_walkers(nwalk):
			"""Initialize walkers around a best initial point"""
			if args.parfile:
				startpars= loadtxt(args.parfile)
				out = analytic_fit.bestFitParameters(startpars)
				best,cov = out[:2]
				print analytic_fit.parameterFitness(best)
				print best[3*nplanets:].reshape(-1,2)
				bestfit = fit(best)
				print best[:3*nplanets].reshape(-1,3)
				print best[3*nplanets:].reshape(-1,2)
				print bestfit
			else:
				bestfit = -inf
				periods = analytic_fit.periodEstimates
				t0s = analytic_fit.tInitEstimates
				Lvals =  -2 * pi/periods * ( t0s - t0s[0])
				PersAndLs = vstack(( periods/periods[0], Lvals  )).T[1:]
				count = 0
				while bestfit == -inf:
			
					randStart = array([(1.e-5,0.02 * random.randn(), 0.02 * random.randn()) for i in range(analytic_fit.nPlanets)])
					randStart = append( randStart , PersAndLs.reshape(-1) )

					out = analytic_fit.bestFitParameters(randStart)
					best,cov = out[:2]
					#best[3*nplanets+1::2] = mod(best[3*nplanets+1::2]+pi,2*pi)-pi
					bestfit = fit(best)
					count+=1
					assert count<101, "Can't find a valid start point!!!"

			return random.multivariate_normal(best,cov/(50.),nwalk)

#----------------------------------------------------------------------

#-----------------------------------------------------------------
#	Set up sampler and walkers
#-----------------------------------------------------------------	
	means=[]
	lnlike = None
	old_best= None
	old_best_lnlike = None
	reset = False
	Nmeasured = 0

	if restart:
		# Read in old walkers
		print "Loading chain from file..."
		p = loadtxt('chain.dat.gz')[-nwalkers:,:]
		lnlike = loadtxt('chain.lnlike.dat.gz')[-nwalkers:]
		print p.shape,lnlike.shape
		old_best= p[argmax(lnlike)]
		old_best_lnlike = fit(old_best)
		print "%d x %d chain loaded"%p.shape
		print "Best likelihood: %.1f"%old_best_lnlike
		
	else:
		# Initialize new walkers
		if nbody:
			p=initialize_walkers(nwalkers,pars0)
		else:
			print "Looking for a good starting point..."
			p=initialize_walkers(nwalkers)

		for x in p.reshape(-1,ndim):
			assert fit(x) > -inf, "Bad IC generated!"
		
	
	# initialize sampler
	sampler = emcee.EnsembleSampler(nwalkers,ndim,fit,threads=nthreads)

	print "Beggining ensemble evolution"
	print "Running with %d parallel threads" % nthreads
	sys.stdout.flush()


	if not restart or args.erase:
		with gzip.open('chain.dat.gz', 'w') as out:
				out.write("# Parameter Chains")
		with gzip.open('chain.lnlike.dat.gz', 'w') as out:
				out.write("# Likelihoods")

	acorstring = ('\t'.join(["M%d\tEX%d\tEY%d"%(d,d,d) for d in range(nplanets)]))+'\t'+('\t'.join(["P%d\tdL%d"%(d,d) for d in range(1,nplanets)]))
#------------------------------------------------
# --- Burn-in Phase and Minimum Search --- #
#------------------------------------------------
	
	if not restart and not args.noloop:
	# If starting MCMC for first time, generate some samples to find a starting place from
		print "Starting burn-in..."
		for p,lnlike,blobs in sampler.sample(p, iterations=nburn, storechain = True):
			pass	
		print "Burn-in complete, starting main loop"

		old_best = sampler.flatchain[argmax(sampler.flatlnprobability)]
		old_best_lnlike = fit(old_best)
	
	if (not restart or args.erase) and nbody and not args.noloop:
	# If starting N-body MCMC for the first time or if the old chains are going to be erased,
	# try to use Levenberg-Marquardt to find the best parameters to start new walkers around
	#
		shrink = 20.
		print "Starting L-M least-squares from likelihood: %.1f"%old_best_lnlike
		out=nbody_fit.CoplanarParametersTTVFit(old_best)
		bestfit,cov = out[:2]

		if fit(bestfit) > old_best_lnlike:
			print "Max likelihood via L-M least-squares: %.1f"%fit(bestfit)
			p = random.multivariate_normal(bestfit,cov/(shrink*shrink),size=nwalkers)
		else:
			print "L-M least-square couldn't find a minimum to initialize around... Returned:"
			print fit(bestfit)
			print "Initializing from the best walkers so far..."
			if not restart:
				p = sampler.flatchain[argsort(-sampler.flatlnprobability)[:nwalkers]] 
		
	sampler.reset()
#------------------------------------------------
# MAIN LOOP
#------------------------------------------------
		
	nloops = int(ceil(nensembles/nthin))
	for k in range(nloops):
		if args.noloop:
			break
		# take 'nthin' samples.
		for p,lnlike,blobs in sampler.sample(p,iterations=nthin, storechain = True):
			pass	

		print '(%d/%d) acceptance fraction = %.3f'%( k+1, nloops, mean(sampler.acceptance_fraction) )
		print 'Autocorrelation lengths: '
		print acorstring
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
			
			#continue

		# Append current state to chain file
		with gzip.open('chain.dat.gz', 'a') as out:
			#print 'appending to chain: ',p.shape
			savetxt(out, p)
		with gzip.open('chain.lnlike.dat.gz', 'a') as out:
			savetxt(out, lnlike)
