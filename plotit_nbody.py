import triangle
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as pl
parser = ArgumentParser(description='Make plots from the output of ensemble sampler with N-body likelihood calculations')
parser.add_argument('--minlnlike','-L', metavar='F', type=float, default= -np.inf, help='Minimum log-likelihood of chain values to include in plot. Default setting includes all values.')
parser.add_argument('--file','-f',metavar='FILE',default=None, help='Save the generated plot as FILE')
parser.add_argument('--burnin',metavar='F',type=float,default=0.2, help='Discard the first F fraction of the chain as burn-in')

args = parser.parse_args()
minlnlike = args.minlnlike
burn_frac = args.burnin

lnlike,chain = np.loadtxt('./chain.lnlike.dat.gz'),np.loadtxt('./chain.dat.gz')
rot = np.array([[0.,1.,0.,.0],[-1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,-1.0,0.]])
evals = np.array([rot.dot(x) for x in chain[:,(1,2,4,5)]])

def linefit(x,y):
	""" 
	Fit a line to data and return intercept and slope.
	"""
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	const = np.ones(len(y))
	A = np.vstack([const,x]).T
	return np.linalg.lstsq(A,y)[0]

innerttvs,outerttvs = np.loadtxt('inner.ttv'),np.loadtxt('outer.ttv')
pin = linefit(innerttvs[:,0],innerttvs[:,1])[1]
pout = linefit(outerttvs[:,0],outerttvs[:,1])[1]

j  = np.argmin( [ np.abs((j-1) * pout / (j*pin) -1.) for j in range(2,7) ] ) + 2
#print pin,pout,j

def Zfn(evals):
	""" Convert individual eccentricities to combined eccentricity entering into first-order MMR potential terms"""
	ex,ey,ex1,ey1 = evals
	if j==2:
		fCoeff = -1.19049
		f1Coeff = 0.42839
	elif j==3:
		fCoeff = -2.02522
		f1Coeff = 2.48401
	elif j==4:
		fcoeff,f1Coeff = -2.84043, 3.28326
	else:
		raise Exception("Laplace coefficients missing!",j)
		
	Zx = fCoeff * ex + f1Coeff * ex1
	Zy = fCoeff * ey + f1Coeff * ey1
	return Zx,Zy, np.linalg.norm( np.array((Zx,Zy)) )

Zdat = np.array([Zfn(edat) for edat in evals])

nburn = int( np.round( burn_frac * len(chain) ) )
chain,Zdat,lnlike  = chain[nburn:],Zdat[nburn:],lnlike[nburn:]

triangle.corner(np.hstack( ( chain[:,(0,3)],Zdat ) )[lnlike > minlnlike] , labels = ('m','m1','Zx','Zy','|Z|') )#,extents = extnts) 

pl.show()

if args.file:
	pl.savefig(args.file)