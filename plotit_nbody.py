import os
import sys
who =os.popen("whoami") 
if who.readline().strip() =='samuelhadden':
	print "On laptop..."
	TTVFAST_PATH = "/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface"
else:
	print "On Quest..."
	TTVFAST_PATH = "/projects/b1002/shadden/7_AnalyticTTV/03_TTVFast/PyTTVFast"
who.close()
sys.path.insert(0,TTVFAST_PATH)
import triangle
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from emcee.autocorr import *

parser = ArgumentParser(description='Make plots from the output of ensemble sampler with N-body likelihood calculations')
parser.add_argument('--minlnlike','-L', metavar='F', type=float, default= -np.inf, help='Minimum log-likelihood of chain values to include in plot. Default setting includes all values.')
parser.add_argument('--file','-f',metavar='PREFIX',default=None, help='Save the generated plots as PREFIX_[plot type].png')
parser.add_argument('--noPlots',default=False, action='store_true' ,help='Don\'t make plots')
parser.add_argument('--truths',metavar='FILE',default=None ,help='File containing the true masses and eccentricity components to show on triangle plot')
parser.add_argument('--extents',metavar='FILE',default=None ,help='File containing parameter extents to show in triangle plot')
parser.add_argument('--analytic',default=False, action='store_true' ,help='Show plots using analytic TTV approximation')
parser.add_argument('--dryrun',default=False, action='store_true' ,help='Load functions only')

args = parser.parse_args()
minlnlike = args.minlnlike

if args.analytic:
	import  AnalyticMultiplanetFit as ttv
else:
	import PyTTVFast as ttv

with open('planets.txt','r') as fi:
	infiles = [ line.strip() for line in fi.readlines()]
input_data= []
for file in infiles:
	input_data.append( np.loadtxt(file) )
nplanets = len(input_data)
if args.analytic:
	fit = ttv.MultiplanetAnalyticTTVSystem(input_data)
else:
	fit = ttv.TTVFitnessAdvanced(input_data)

def linefit(x,y):
	""" 
	Fit a line to data and return intercept and slope.
	"""
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	const = np.ones(len(y))
	A = np.vstack([const,x]).T
	return np.linalg.lstsq(A,y)[0]

def dataMedianAndRange(data,q=90.0):
	med = np.median(data)
	dq = 0.5 * (100. - q)
	lo = np.percentile(data, dq)
	hi = np.percentile(data,100. - dq)
	return (med,hi-med,lo-med)

def ConfidenceRegions2D(x,y,*args,**kwargs):
	"""
	Plot a 2-D histogram of samples.
	"""
	ax = kwargs.pop("ax", pl.gca())
	extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
	bins = kwargs.pop("bins", 50)
	color = kwargs.pop("color", "k")
	cmap = kwargs.pop("cmap",cm.get_cmap("Blues"))
	label = kwargs.pop("label",None)
	levels = np.hstack(( [0.0] , kwargs.pop("levels",[0.68]) )) 

	X = np.linspace(extent[0][0], extent[0][1], bins + 1)
	Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
	try:
	    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y),weights=kwargs.get('weights', None))
	except ValueError:
		raise ValueError("It looks like at least one of your sample columns have no dynamic range. You could try using the `extent` argument.")

	V = levels
	Hflat = H.flatten()
	inds = np.argsort(Hflat)[::-1]
	Hflat = Hflat[inds]
	sm = np.cumsum(Hflat)
	sm /= sm[-1]

	for i, v0 in enumerate(V):
	    try:
	        V[i] = Hflat[sm <= v0][-1]
	    except:
	        V[i] = Hflat[0]

	X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
	X, Y = X[:-1], Y[:-1]

	#,colors=cmap(1.)
	cols = [cmap(1.0) for lvl in levels]
	ax.contour(X1, Y1, H.T,V[::-1],colors=cols, linestyles='-',label=label)
	ax.contourf(X1, Y1, H.T,V[::-1],cmap = cmap,alpha=kwargs.pop("alpha",1))

	ax.set_xlim(extent[0])
	ax.set_ylim(extent[1])
	return V[::-1]

if not args.dryrun:

  with open('planets.txt','r') as fi:
	  plfiles = [line.strip() for line in fi.readlines()]

  nplanets = len(plfiles)
  ttvs = []
  periods = np.zeros(nplanets)
  lnlike,chain = np.loadtxt('./chain.lnlike.dat.gz'),np.loadtxt('./chain.dat.gz')
		 

  rot = np.array([[0.,1.,0.,.0],[-1.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,-1.0,0.]])
  e_indices =  np.array([ (3*n+1,3*n+2) for n in range(nplanets) ]).reshape(-1)
  mass_indices =  tuple([ 3*n for n in range(nplanets) ])


  evals = np.array([x for x in chain[:,tuple(e_indices)] ])
  masses=chain[:,mass_indices]

  for i,file in enumerate(plfiles):
	  print file
	  ttvs.append( np.loadtxt(file) )
	  periods[i] = linefit((ttvs[-1])[:,0] , (ttvs[-1])[:,1])[1]
	  print periods[i]

  def reshapeChain(walkerPosition):
	  massvals = walkerPosition[mass_indices,]
	  evals = walkerPosition[e_indices,].reshape(-1,2)
	  rMatrix = np.array([[0.,1.],[-1.,0]])
 
	  if args.analytic:
		  return np.append(massvals,(evals).reshape(-1)) 
	  else:
		  evec = np.array([np.dot(rMatrix,v) for v in evals]).reshape(-1)
		  return np.append(massvals,evec) 

  def get_j_delta(p1,p2):
	  pratio = np.min((p2/p1,p1/p2))
	  j = np.argmin([abs((j-1)*pratio/j-1) for j in range(2,6)]) + 2
	  delta = (j-1.) * pratio / j - 1.
	  return j,delta

  def Zfn(evals,j):
	  """Convert individual eccentricities to combined eccentricity entering into first-order MMR potential terms"""
	  if args.analytic:
		  ex,ey,ex1,ey1 = evals
	  else:
		  ex,ey,ex1,ey1 = rot.dot(evals)
	  if j==2:
		  fCoeff = -1.19049
		  f1Coeff = 0.42839
	  elif j==3:
		  fCoeff = -2.02522
		  f1Coeff = 2.48401
	  elif j==4:
		  fCoeff,f1Coeff = -2.84043, 3.28326
	  elif j==5:
		  fCoeff,f1Coeff = -3.64962, 4.08371
	  elif j==6:
		  fCoeff,f1Coeff = -4.45614, 4.88471
	  else:
		  raise Exception("Laplace coefficients missing!",j)
	 
	  Zx = fCoeff * ex + f1Coeff * ex1
	  Zy = fCoeff * ey + f1Coeff * ey1
	  return Zx,Zy, np.linalg.norm( np.array((Zx,Zy)) )
  true_fit=None
  if args.truths:
	  truths = np.loadtxt(args.truths)
	  if not args.analytic:
		  true_fit = fit.CoplanarParametersFitness(truths)
		  print true_fit
	  truths = reshapeChain(truths)
	  
  else:
	  truths = None

  Zdat = []
  for i in range(nplanets-1):
	  j = get_j_delta(periods[i],periods[i+1])[0]
	  eccvecs = evals[:,(i,i+1,i+2,i+3)]
	  Zdat.append( np.array([Zfn(eccs,j) for eccs in eccvecs ]) )


  best,best_lnlike= chain[np.argmax(lnlike)],np.max(lnlike)
  with open("summary.txt",'w') as fi:
	  fi.write("Best fit parameters: (log-likelihood=%.1f)\n "%best_lnlike )
	  best_string = "\t".join(map(lambda x: "%.4g"%x ,best))
	  fi.write(best_string)
	  fi.write('\n\n')
	  for i in range(nplanets):
		  fi.write("Planet %d:\n"%i)
		  fi.write("(Median,+,-)\n")
		  fi.write("\t68 pct. Mass (x 1E6): %.1f^{%+.1f}_{%+.1f}\n"%dataMedianAndRange( masses[:,i]*1e6, q = 68.))
		  fi.write("\t95 pct. Mass (x 1E6): %.1f^{%+.1f}_{%+.1f}\n"%dataMedianAndRange( masses[:,i]*1e6, q = 95. ))
		  fi.write("\t68 pct. ex: %+.3f^{%+.3f}_{%+.3f}\n"%dataMedianAndRange( evals[:,2*i], q = 68. ))
		  fi.write("\t95 pct. ex: %+.3f^{%+.3f}_{%+.3f}\n"%dataMedianAndRange( evals[:,2*i], q = 95. ))
		  fi.write("\t68 pct. ey: %+.3f^{%+.3f}_{%+.3f}\n"%dataMedianAndRange( evals[:,2*i+1], q = 68.))
		  fi.write("\t95 pct. ey: %+.3f^{%+.3f}_{%+.3f}\n"%dataMedianAndRange( evals[:,2*i+1], q = 95. ))
  np.savetxt("bestpars.txt",best)

  plotData = np.array([reshapeChain(x) for x in chain[lnlike>minlnlike]])
  minMass = np.min( plotData[:,:3].reshape(-1) )
  maxMass = np.max( plotData[:,:3].reshape(-1) )

  if args.extents:
	  with open(args.extents) as fi:
		  lines = [map(float,line.split()) for line in fi.readlines() ]
	  extnts = [x[0] if len(x)==1 else x for x in lines]
  else:
	  maxEcc = np.max( np.abs(plotData[:,3:].reshape(-1) ))
	  extnts = [ ( np.min(plotData[:,i]), np.max(plotData[:,i]) ) for i in range(nplanets) ] + [ (-maxEcc,maxEcc) for i in range(2*nplanets)]
	  with open("plot_ranges.txt","w") as fi:
		  for val in extnts:
			  fi.write(  "\t".join(map(str,val)) )
			  fi.write("\n")
  if not args.noPlots:
  #-- Lnlike histogram --#
	  pl.hist(lnlike[lnlike>minlnlike],bins=100)
	  if args.analytic:
		  ntrTot=np.sum(np.array(map(len,fit.transitTimes)))
	  else:
		  ntrTot=np.sum(np.array(map(len,fit.transit_times)))
	  pl.title('N_transits/2 = %.2f'%(0.5 * ntrTot ))
	  if true_fit:
		  pl.axvline(true_fit)

	  if args.file:
		  pl.savefig("%s_chi2.png"%args.file)
 
  #-- Pair-wise mass versus Z plots --#
	  for i in range(nplanets-1):
		  triangle.corner(np.hstack( ( chain[:,(3*i,3*(i+1))],Zdat[i] ) )[lnlike > minlnlike] , labels = ('m','m1','Zx','Zy','|Z|') ,quantiles=[0.16,0.5,0.84])
		  if args.file:
			  pl.savefig("%s_mass-vs-Z_%d.png"%(args.file,i))
 

  #-- Full mass versus ecc corner plot --#
	  lbls = [['m%d'%i] for i in range(nplanets)] + [('ex%d'%i,'ey%d'%i) for i in range(nplanets)] 
	  flatlabels =[]
	  for lbl in lbls:
		  for y in lbl:
			  flatlabels.append(y)

	  lbls = tuple(flatlabels)

	  triangle.corner(plotData , labels = lbls ,truths=truths,extents=extnts,quantiles=[0.16,0.5,0.84])

	  if args.file:
		  pl.savefig("%s_mass-vs-ecc.png"%args.file)
 
  #-- Best-fit TTV plot --#
	  if args.analytic:
		  pl.figure()
		  fit.parameterTTVPlot(best)
	  else:
		  fit.CoplanarParametersTTVPlot(best)
	  if args.file:
		  pl.savefig("%s_ttvs_best.png"%args.file)
  #-- Best-fit TTV resdiuals plot --#
	  if args.analytic:
		  pl.figure()
		  fit.parameterTTVResidualsPlot(best)
	  else:
		  fit.CoplanarParametersTTVResidPlot(best)
	  if args.file:
		  pl.savefig("%s_ttvResids_best.png"%args.file)
  #
	  pl.show()
		
 
