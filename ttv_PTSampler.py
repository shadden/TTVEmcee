#TTVFAST_PATH = "/projects/b1002/shadden/7_AnalyticTTV/03_TTVFast/PyTTVFast"
TTVFAST_PATH = "/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface"

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

def write_headers(Temps):
	for i,temp in enumerate(Temps):
		header = "# Chain {0:02d} of {1:02d}, temperature: {2:06.3f}\n".format(i,len(Temps),temp)
		with gzip.open('chain.{0:02d}.dat.gz'.format(i), 'w') as out:
			out.write(header)
            	with gzip.open('chain.{0:02d}.lnlike.dat.gz'.format(i), 'w') as out:
                	out.write(header)
            	with gzip.open('chain.{0:02d}.lnpost.dat.gz'.format(i), 'w') as out:
                	out.write(header)
#------------------------------------------
#  MAIN
#------------------------------------------
if __name__=="__main__":

	parser = ArgumentParser(description='run an Parallel temperature MCMC analysis of a system of planet\'s TTVs')
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
	parser.add_argument('--ntemps', metavar='N', type=int, default=4, help='number different temperature scales to use')

	
	parser.add_argument('--input','-I',metavar='FILE',default='planets.txt',help='File that lists the names of the files containing input transits')
	
#----------------------------------------------------------------------------------

	args = parser.parse_args()
	restart = args.restart
	nensembles=args.nensembles
	nwalkers = args.nwalkers
	nthin=args.nthin
	nthreads=args.nthreads
	nburn = args.nburn
	infile = args.input
	
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
		pars0 = append(ones(nplanets) * 1.e-5 , zeros(2*nplanets) )
#----------------------------------------------------------------------		
# Priors and likelihood functions
#----------------------------------------------------------------------		

	sys.path.insert(0,TTVFAST_PATH)
	import PyTTVFast as ttv
	ndim = 5*nplanets-2
	nbody_fit = ttv.TTVFitnessAdvanced(input_data)

	def logp(x):
		# Masses must be positive
		masses = (x[::3])[:nplanets]
		bad_masses = any(masses < 0.0)
		if bad_masses:
			return -inf
		
		# Mean anomalies lie between -pi and pi
		meanAnoms = x[3*nplanets+1::2]
		bad_angles = any( meanAnoms > pi)
		if bad_angles:
			return -inf
		
		# Eccentricities must be smaller than 1
		exs,eys =(x[1::3])[:nplanets],(x[2::3])[:nplanets]
		bad_eccs = any(exs**2 +eys**2 >= 0.9**2)
		if bad_eccs:
			return -inf
		# If parameters are acceptable, return 0.
		return 0.
		
	def fit(x):
		
		return nbody_fit.CoplanarParametersFitness(x)

	def initialize_walkers(nwalk,p0):
		"""Initialize walkers around point p0 = [ mass1,mass2,...,ex1,ey1,ex2,ey2,...,P2_obs/P1_obs,..., dL2_obs,...]"""
		masses = p0[:nplanets]
		evecs = p0[nplanets:].reshape(-1,2)
		ics = nbody_fit.GenerateRandomInitialConditions(masses,0.1,evecs,0.002,nwalk)
		return array([ nbody_fit.convert_params(ic) for ic in ics])

	means=[]
	p = zeros((ntemps,nwalkers, ndim))
		
	acorstring = ('\t'.join(["M%d\tEX%d\tEY%d"%(d,d,d) for d in range(nplanets)]))+'\t'+('\t'.join(["P%d\tdL%d"%(d,d) for d in range(1,nplanets)]))

#-----------------------------------------------------------------
#	Set up sampler and walkers
#-----------------------------------------------------------------	

	if restart:
		# Read in old walkers
		for i in range(ntemps):
			print "Loading chain %02d from file..."%i
			p[i, :, :] = loadtxt('chain.%02d.dat.gz'%i)[-nwalkers:,:]
	else:
		# Initialize new walkers
		p=[initialize_walkers(nwalkers,pars0)]
      	for temp in range(ntemps-1):
        	p.append(initialize_walkers(nwalkers,pars0))
	
	for x in p.reshape(-1,ndim):
		assert fit(x) > -inf and logp(x)> -inf, "Bad IC generated!"
		
	
	# initialize sampler
	sampler = sampler = emcee.PTSampler(ntemps,nwalkers,ndim,fit,logp,threads=nthreads)
	Ts = 1.0/sampler.betas
	if not restart and not args.noloop:
		write_headers(Ts)

	print "Beggining ensemble evolution"
	print "Running with %d parallel threads" % nthreads
	print
#-----------------------------------------------------------------	
	sys.stdout.flush()
	lnpost = None
	lnlike = None
	old_best_lnlike = None
	reset = False
	Nmeasured = 0


#------------------------------------------------
# MAIN LOOP
#------------------------------------------------
	# --- Burn-in Phase --- #
	if not restart and not args.noloop:
		print "Starting burn-in..."
		for  p,lnpost,lnlike in sampler.sample(p, iterations=nburn, storechain = False):
			pass	
		for i in range(ntemps):
			fchain = sampler.flatchain[i]
			best_indices = argsort(lnlike[i].reshape(-1,))[:nwalkers]
			p[i] = fchain[best_indices] 
		
		sampler.reset()

		print "Burn-in complete, starting main loop"

	nloops = int(ceil(nensembles/nthin))

	for k in range(nloops):
		if args.noloop:
			break
		# take 'nthin' samples.
		for p,lnpost,lnlike in sampler.sample(p,iterations=nthin, storechain = True):
			pass	

		print 'acceptance fraction = ', ' '.join( map( lambda x: '{0:6.3f}'.format(x), mean(sampler.acceptance_fraction,axis = 1))) 
		print 'Tswap fraction      = ', ' '.join(map(lambda x: '{0:6.3f}'.format(x), sampler.tswap_acceptance_fraction))
		print 'Autocorrelation lengths: '
		print acorstring
		for taus in sampler.acor():
			print '\t'.join(map(lambda x: '{0:.1f}'.format(x), taus))

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
		
		
		
		for i in range(ntemps):
			with gzip.open('chain.{0:02d}.dat.gz'.format(i), 'a') as out:
				savetxt(out, p[i,:,:])
            with gzip.open('chain.{0:02d}.lnlike.dat.gz'.format(i), 'a') as out:
            	savetxt(out, lnlike[i,:].reshape((1,-1)))
            with gzip.open('chain.{0:02d}.lnpost.dat.gz'.format(i), 'a') as out:
            	savetxt(out, lnpost[i,:].reshape((1,-1)))

