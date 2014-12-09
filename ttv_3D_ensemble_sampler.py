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
import inclinations
from scipy.optimize import minimize

def mod_angvars(p,nplanets):
	
	p[ tuple([ 4+i*7 for i in range(nplanets)]), ] = mod( p[ tuple([ 4+i*7 for i in range(nplanets)]), ] + pi ,2*pi ) - pi
	p[ tuple([ 5+i*7 for i in range(nplanets)]), ] = mod( p[ tuple([ 5+i*7 for i in range(nplanets)]), ] + pi ,2*pi ) - pi
	return p
def convert2rel_node(par,nplanets):
	node_dex = tuple([5+i*7 for i in range(1,nplanets)])
	par[node_dex,] -= par[5]
	par[node_dex,] = mod( par[node_dex,] + pi, 2*pi ) - pi
	return append(par[:5],par[6:])
#------------------------------------------
#  MAIN
#------------------------------------------
if __name__=="__main__":

	parser = ArgumentParser(description='run an ensemble MCMC analysis of a pair of TTVs')

	parser.add_argument('--restart', default=False, action='store_true', help='continue a previously-existing run')
	parser.add_argument('--erase', default=False, action='store_true', help='Start walkers from old files but overwrite them')

	parser.add_argument('-n','--nensembles', metavar='N', type=int, default=100, help='number of ensembles to accumulate')
	parser.add_argument('--nwalkers', metavar='N', type=int, default=100, help='number of walkers to use')	
	parser.add_argument('--nthin', metavar='N', type=int, default=10, help='number of steps to take between each saved ensemble state')
	parser.add_argument('--nburn', metavar='N', type=int, default=100, help='Number of burn-in steps')
	parser.add_argument('--nthreads', metavar='N', type=int, default=multi.cpu_count(), help='number of concurrent threads to use')

	parser.add_argument('-P','--parfile', metavar='FILE', default=None, help='Text file containing parameter values to initialize walker around.')
	parser.add_argument('--noloop', default=False, action='store_true', help='Run set-up but do not excecute the MCMC main loop')
	parser.add_argument('--input','-I',metavar='FILE',default='planets.txt',help='File that lists the names of the files containing input transits')
	parser.add_argument('--priors',metavar='[g | l]',default=None,help='Use eccentricity priors. g: Gaussian , l: log-uniform')

	parser.add_argument('--relative_coords',default=False,action='store_true',help='Reduce number of parameter dimensions by fixing ascending node of inner planet to 0')
	parser.add_argument('--coplanar',default=False,action='store_true',help='Model TTVs with coplanar planets.')
	
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
	priors = args.priors
	
	rel_nodes = args.relative_coords
	coplanar = args.coplanar
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
#----------------------------------------------------------------------------------


#--------------------------------------------
# Set up a TTVFit object for likelihood computations
	sys.path.insert(0,TTVFAST_PATH)
	import PyTTV_3D as ttv
	
	if rel_nodes:
		ndim = 7*nplanets - 1
	elif coplanar:
		ndim = 5*nplanets
	else:
		ndim = 7*nplanets 
	
	nbody_fit = ttv.TTVFit(input_data)
#--------------------------------------------

#--------------------------------------------
# Impact parameter information
	with open("inclination_data.txt") as fi:
		lines = [l.split() for l in fi.readlines()]
	mstar,sigma_mstar = map(float,lines[0])
	rstar,sigma_rstar = map(float,lines[1])
	b,sigma_b = array([map(float,l[1:]) for l in lines[2:] ]).T
	
	b_Obs = ttv.ImpactParameterObservations([rstar,sigma_rstar],[mstar,sigma_mstar], vstack((b,sigma_b)).T)
#--------------------------------------------
	def logpInc(x):
		if rel_nodes:
			xs = insert(x,5,0.).reshape(-1,7)
		elif coplanar:
			return 0
		else:
			xs = x.reshape(-1,7)
		inclinations = xs[:,4]
		periods = xs[:,1]
		return b_Obs.ImpactParametersPriors(inclinations, periods) 

#--------------------------------------------
# Likelihood function

	def fit(x):
		if rel_nodes:
			xs = insert(x,5,0.).reshape(-1,7)
		elif coplanar:
			xs = x.reshape(-1,5)
		else:
			xs = x.reshape(-1,7)
	
		# Masses must be positive			
		masses = xs[:,0]
		bad_masses = any(masses < 0.0)
		if bad_masses:
			return -inf

		# Eccentricities must be smaller than 1
		exs,eys = xs[:,(2,3)].T
		bad_eccs = any(exs**2 +eys**2 >= 0.9**2)
		if bad_eccs:
			return -inf
		
		if priors=='g':
			logp = -1.0*sum( 0.5 * (exs**2 + eys**2 ) / 0.017**2 )
		elif priors =='l':
			logp = np.sum( log10( 1.0 / sqrt(exs**2 + eys**2) ) )
		else:
			logp = 0.0
		
		if coplanar:
			return nbody_fit.ParameterFitness(x) + logp
		
		# Angles should be between -pi and pi.  
		#  Only evaluated if not coplanar model
		angs = xs[:,(4,5)].reshape(-1)
		bad_angs = any(abs(angs) > pi )
		if bad_angs:
			return -inf
			
		return nbody_fit.ParameterFitness(x) + logp	+ logpInc(x)			
#--------------------------------------------

#-----------------------------------------------------------------
#	# --- Initialize Walkers  --- #
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
		lnlike = loadtxt('chain.lnlike.dat.gz')
		if args.erase:
			# take the best old walker positions
			lnlike,nlnlike = sort(lnlike)[-nwalkers:] ,argsort(lnlike)[-nwalkers:]
			p = loadtxt('chain.dat.gz')[nlnlike,:]
		else:
			lnlike = lnlike[-nwalkers]
			p = loadtxt('chain.dat.gz')[-nwalkers,:]
		print p.shape,lnlike.shape
		old_best= p[argmax(lnlike)]
		old_best_lnlike = fit(old_best)
		print "%d x %d chain loaded"%p.shape
		print "Best likelihood: %.1f"%old_best_lnlike
	
	elif args.parfile:
		old_best = pars0
		old_best_lnlike = fit(old_best)
	
	else:
		# Initialize new walkers
		ic = nbody_fit.coplanar_initial_conditions(.3e-5*ones(nplanets),random.normal(0,0.01,nplanets),random.normal(0,0.01,nplanets))
		fitdata= nbody_fit.LeastSquareParametersFit( ic[:,(0,1,2,3,6)] )
		best,cov = fitdata[:2]
		
		print "Initial (coplanar) Fitness: %.2f"%nbody_fit.ParameterFitness(best)
		if not coplanar:
			# 3D L-M Fit
			best3d = best.reshape(-1,5)
			i0,sigma_i=b_Obs.ImpactParametersToInclinations(nbody_fit.Observations.PeriodEstimates)
			best3d = hstack(( best3d[:,:4], i0.reshape(-1,1) , random.uniform(-0.005,0.005,(nplanets,1)), best3d[:,-1].reshape(-1,1) ))
			best3d = best3d.reshape(-1)
			best3d[tuple([0+7*i for i in range(nplanets)]),] = abs(best3d[tuple([0+7*i for i in range(nplanets)]),])

			fitdata = nbody_fit.LeastSquareParametersFit( best3d )
			best,cov = fitdata[:2]
			best = mod_angvars(best,nplanets)

			if rel_nodes:
				print "Initial 3D Fitness: %.2f"%fit(convert2rel_node(best,nplanets))
			else:
				print "Initial 3D Fitness: %.2f"%fit(best)
	
			fitdata = nbody_fit.LeastSquareParametersFit( best , [cos(i0), diagonal(sigma_i) ])
			best,cov = fitdata[:2]
			best = mod_angvars(best,nplanets)
		
			if rel_nodes:
				print "3D Fitness: %.2f"%fit(convert2rel_node(best,nplanets))
			else:
				print "3D Fitness: %.2f"%fit(best)

			for par in best.reshape(nplanets,-1):
				print "\t",par	
	
		shrink = 10.
		p = zeros(( nwalkers,ndim ))
		for i in range(nwalkers):
			par = random.multivariate_normal(best,cov/(shrink*shrink))
			if not coplanar:
				par = mod_angvars(par,nplanets)
				# randomly flip I's to alternate value that gives the same impact parameter
				for j in range(nplanets):
					if random.choice([True,False]):
						par[j*7 + 4] = pi - par[j*7 + 4]
				if rel_nodes:
					par = convert2rel_node(par,nplanets)

			#---- If initial parameter vector draw is bad, draw until a good vector is initialized ----#
			while fit(par) == -inf:
				par = random.multivariate_normal(best,cov/(shrink*shrink))
				if not coplanar:
					par = mod_angvars(par,nplanets)
					# randomly flip I's to alternate value that gives the same impact parameter
					for j in range(nplanets):
						if random.choice([True,False]):
							par[j*7 + 4] = pi - par[j*7 + 4]
					if rel_nodes:
						par = convert2rel_node(par,nplanets)

			p[i] = par		
				
	# initialize sampler
	sampler = emcee.EnsembleSampler(nwalkers,ndim,fit,threads=nthreads)

	print "Beggining ensemble evolution"
	print "Running with %d parallel threads" % nthreads
	sys.stdout.flush()


	if not restart or args.erase:
		with gzip.open('chain.dat.gz', 'w') as out:
				out.write("# Parameter Chains\n")
		with gzip.open('chain.lnlike.dat.gz', 'w') as out:
				out.write("# Likelihoods\n")

#------------------------------------------------
# --- Burn-in Phase --- #
#------------------------------------------------
	
	if not restart and not args.noloop and not args.parfile:
	# If starting MCMC for first time, generate some samples to find a starting place from
		print "Starting burn-in..."
		for p,lnlike,blobs in sampler.sample(p, iterations=nburn, storechain = True):
			pass
		print "Burn-in complete, starting main loop"

		old_best = sampler.flatchain[argmax(sampler.flatlnprobability)]
		old_best_lnlike = fit(old_best)
	
	sampler.reset()
#------------------------------------------------
# --- Main Loop  --- #
#------------------------------------------------
		
	nloops = int(ceil(nensembles/nthin))
	for k in range(nloops):
		if args.noloop:
			break
		# take 'nthin' samples.
		for p,lnlike,blobs in sampler.sample(p,iterations=nthin, storechain = False):
			pass
		
		print '(%d/%d) acceptance fraction = %.3f'%( k+1, nloops, mean(sampler.acceptance_fraction) )
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
			savetxt(out, p)
		with gzip.open('chain.lnlike.dat.gz', 'a') as out:
			savetxt(out, lnlike)
