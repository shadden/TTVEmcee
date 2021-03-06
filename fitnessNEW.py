from numpy import *

import sys
from itertools import combinations
sys.path.insert(0, '/Users/samuelhadden/13_HighOrderTTV/TTVEmcee')

import LaplaceCoefficients as LC
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#---------------------------------------------------
# miscellaneous
#---------------------------------------------------
def linefit(x,y):
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	const = np.ones(len(y))
	A = np.vstack([const,x]).T
	return np.linalg.lstsq(A,y)[0]

def linearfit(x,y):
      assert len(x)==len(y), "length of arrays unequal! x: {}, y: {}".format(len(x),len(y))
      a = array([ones(len(x)),x]).T
      return linalg.lstsq(a,y)[0]

def fullPeriodFit(trdata1,trdata2,precision=1.e-5,itermin=2,itermax = 10):
	n,t =  trdata1.T[:2]
	n1,t1 = trdata2.T[:2]
	p,p1 = map( lambda x: linearfit(x[0],x[1])[1], [[n,t],[n1,t1]] )
	print p,p1
#
	iter = 0
	dpMax = 1.
	while ( dpMax > precision and iter < itermax ) or iter < itermin :
		A = array([ones(len(n)),n])
		A1 = array([ones(len(n1)),n1])
		for i in range(1,5):
			s = sin( i * 2*pi * t / p1 )
			c = cos( i * 2*pi * t / p1 )
			A = append(A,[s,c],axis=0)
			s1 = sin( i * 2*pi * t1 / p )
			c1 = cos( i * 2*pi * t1 / p )
			A1 = append(A1,[s1,c1],axis=0)
		A = A.T
		A1 = A1.T
		fit = linalg.lstsq(A,t)[0]
		fit1 = linalg.lstsq(A1,t1)[0]
		
		t0,pNew = fit[:2]
		t01,p1New = fit1[:2]
		dpMax = max([abs((p-pNew)/pNew),abs((p1-p1New)/p1New)])
		p=pNew
		p1 = p1New
		iter += 1
		print iter,p,p1

	return [p,p1,t0,t01]

def fullPeriodFit2(trdata1,trdata2):
	n,t =  trdata1.T[:2]
	n1,t1 = trdata2.T[:2]
	per0,per10 = map( lambda x: linearfit(x[0],x[1])[1], [[n,t],[n1,t1]] )
	mdl = lambda x,p,t0,a,b,w: x * p + t0 + a * sin(w * x*p) + b * cos(w * x*p)
	j =2 +argmin([abs((j-1)*per10 / ( j*per0 ) - 1.) for j in range(2,5)])
	w0 = 2. * pi * ( j / per10 - (j-1) / per0 )
	popt, pcov = curve_fit(mdl, n, t, p0=[per0,t[0], 0.05,0.05,w0] )
	popt1, pcov1 = curve_fit(mdl, n1, t1, p0=[per10,t1[0], 0.05, 0.05,w0] )

	return popt[0],popt[1], popt1[0], popt1[1]

def TrimData(data,tol=3.0):
	t0s,periods=array([ linearfit(x[:,0],x[:,1])  for x in data ]).T
	
	rms = array([ sqrt( var(tr[:,1] - periods[i] * tr[:,0] - t0s[i] )) for i,tr in enumerate(data)])
	print rms
	trimmed_data = []
	for i,tr in enumerate(data):
		trimmed_data.append( array( [ x for x in tr if abs( x[1] - x[0] *periods[i] -t0s[i] ) < tol * rms[i] ]) )

	return trimmed_data
	
def fullPeriodFit2plot(trdata1,trdata2,resids=False,planets=3):
	"""If the option 'resids' is specified, plot the redisuals of the model fitting procedure.  
	Otherwise show the TTVs plus the best-fit sinusoidal model"""
	n,t =  trdata1.T[:2]
	n1,t1 = trdata2.T[:2]
	per0,per10 = map( lambda x: linearfit(x[0],x[1])[1], [[n,t],[n1,t1]] )
	mdl = lambda x,p,t0,a,b,w: x * p + t0 + a * sin(w * p * x) + b * cos(w * p * x)
	j =2 +argmin([abs((j-1)*per10 / ( j*per0 ) - 1.) for j in range(2,5)])
	w0 = 2. * pi * ( j / per10 - (j-1) / per0 )

	popt, pcov = curve_fit(mdl, n, t, p0=[per0,t[0], 0.05, 0.05, w0 ] )
	popt1, pcov1 = curve_fit(mdl, n1, t1, p0=[per10,t1[0], 0.05, 0.05, w0 ] )
	if resids:
		errs,errs1 = trdata1[:,2],trdata2[:,2]
		if planets==1:
			plt.errorbar(t,t - mdl(n,popt[0],popt[1],popt[2],popt[3],popt[4]),yerr=errs,fmt='ks')
		elif planets==2:
			plt.errorbar(t1,t1 - mdl(n1,popt1[0],popt1[1],popt1[2],popt1[3],popt1[4]) ,yerr = errs1,fmt='rs')
		else:
			plt.errorbar(t,t - mdl(n,popt[0],popt[1],popt[2],popt[3],popt[4]),yerr=errs,fmt='ks')
			plt.errorbar(t1,t1 - mdl(n1,popt1[0],popt1[1],popt1[2],popt1[3],popt1[4]) ,yerr = errs1,fmt='rs')
	
	else:
		plot(n,mdl(n,popt[0],popt[1],popt[2],popt[3],popt[4])- popt[0]*n -popt[1] )
		plot(n,t - popt[0]*n - popt[1] )
		plot(n1,mdl(n1,popt1[0],popt1[1],popt1[2],popt1[3],popt1[4])- popt1[0]*n1 -popt1[1] )
		plot(n1,t1 - popt1[0]*n1 - popt1[1] )
	plt.show()
def delta(pratio,j,k):
	return pratio * k/j - 1.
		
def get_res(pratio):
	return argmin([abs(delta(pratio,j,j-1)) for j in range(2,6)]) + 2

def sinusoids(p,p1,L0,L10,j,k):

	freq = 2. * pi * ( ( j / p1) - ( k / p) ) 
	phi0 = j * L10  - k * L0
	sinfn =  lambda t: sin(freq * t + phi0)
	cosfn =  lambda t: cos(freq * t + phi0)
	
	return array([cosfn,sinfn])
##def time_to_ttv(time):
#	n = arange(len(time))
#	a = array([ones(len(time)),n]).T
#	b,m=linalg.lstsq(a, time)[0]
#	return array(time-m*n-b)
def time_to_ttv(n,time):
	a = array([ones(len(time)),n]).T
	b,m=linalg.lstsq(a, time)[0]
	return array(time-m*n-b)

			
				
			
			

#---------------------------------------------------
# Recenter walkers
#---------------------------------------------------
def recenter_best(chains, best, logprior, shrinkfactor=10.0, nthreads=1):
      """Returns the given chains re-centered about the best point.
            
            :param chains: Shape ``(NTemps, NWalkers, NParams)``.
            
            :param best: The best point from the chain.
            
            :param lnpost: Log-posterior object (used to check prior bounds).
            
            :param shrinkfactor: The shrinkage factor in each dimension with
            respect to the spread of ``chain[0, :, :]``.
            
            """
    
      covar = cov(chains[0,:,:], rowvar = 0)
      covar/= shrinkfactor*shrinkfactor
    
      new_chains = random.multivariate_normal(best, covar, size=chains.shape[:-1])
    
      for i in range(new_chains.shape[0]):
            for j in range(new_chains.shape[1]):
                  while logprior(new_chains[i,j,:]) == float('-inf'):
                        new_chains[i,j,:] = random.multivariate_normal(best, covar)
    
      new_chains[0,0,:] = best
      return new_chains

class fitness(object):
########################################################################
	def __init__(self,input_data,input_data1):
#################
# Orbital parameters
#################
		p,t0,p1,t10 = fullPeriodFit2(input_data[:,:2],input_data1[:,:2])
		self.T0 = t0
		self.T10 = t10
		self.p = p
		self.p1 = p1
		self.pratio = p1/p
		self.alpha = (self.pratio)**(-2./3.)
		self.j = get_res(self.p1/self.p)
#################
# Laplace coefficients
#################
		self.f0 = -0.5 * self.alpha * LC.b(self.alpha,0.5,0,1)
		self.f10  = 1.5 / self.alpha**2 - 0.5 * LC.b(self.alpha,0.5,1,0) + 0.5 * self.alpha * LC.b(self.alpha,0.5,1,1)
#
		self.f = LC.get_f_array(self.pratio)
		self.f1Ext = LC.get_f1Ext_array(self.pratio)
		self.f1Int = LC.get_f1Int_array(self.pratio)
#	
		self.df = LC.get_df_array(self.pratio)
		self.df1Ext = LC.get_df1Ext_array(self.pratio)
		self.df1Int = LC.get_df1Int_array(self.pratio)
#	
		self.kj = LC.get_k_array(self.pratio)
		self.k1j = LC.get_k1_array(self.pratio)
		self.dkj = LC.get_dk_array(self.pratio)
		self.dk1j = LC.get_dk1_array(self.pratio)
#	
		self.g = LC.get_g_array(self.pratio)
		self.g1Ext = LC.get_g1Ext_array(self.pratio)
		self.g1Int = LC.get_g1Int_array(self.pratio)
		self.h = LC.get_h_array(self.pratio)
#################
# Transit Data
#################
		self.input_data = array(input_data)
		self.input_data1 = array(input_data1)	
		self.transits = self.input_data[:,1]
		self.transits1 = self.input_data1[:,1]
		self.trN = self.input_data[:,0]
		self.trN1 = self.input_data1[:,0]
	########################################################################
	########################################################################
	# First-order terms
	########################################################################
	def dz(self,j):
		if j==0:
			denom = sqrt(self.alpha) * self.pratio
			num =  self.f0
			return num / denom
		return -1.0 * self.f[j-1] / ( sqrt(self.alpha)  * j * delta(self.pratio,j,j-1)  )
		
	def dz1(self,j):
		if j==0:
			return self.f10 / self.pratio
		return -1.0  * self.f1Int[j-1] / (j * delta(self.pratio,j,j-1))
	
	def dl(self,j,ex,ey,ex1,ey1):
		if j==0:
			return zeros(2)
		Zx = self.f[j-1] * ex + self.f1Ext[j-1] * ex1
		Zy = self.f[j-1] * ey + self.f1Ext[j-1] * ey1
		dZx = self.df[j-1] * ex + self.df1Ext[j-1] * ex1
		dZy =  self.df[j-1] * ey + self.df1Ext[j-1] * ey1
		coeff = (j-1.) * 3. / (j * 2 * self.alpha**2 * j *  (delta(self.pratio,j,j-1.))**2 )
		coeff2 = sqrt(self.alpha) / (j * delta(self.pratio,j,j-1.))
		# This is correct-- note 1/i factor and Zconj in definition	
		dlx = -1. * coeff * Zy - coeff2 * dZy
		dly = -1. * coeff * Zx - coeff2 * dZx
	
		return array([dlx,dly])
	
		
	def dl1(self,j,ex,ey,ex1,ey1):
		if j==0:
			return zeros(2)
		Zx = self.f[j-1] * ex + self.f1Int[j-1] * ex1
		Zy = self.f[j-1] * ey + self.f1Int[j-1] * ey1
		dZx = self.df[j-1] * ex + self.df1Int[j-1] * ex1
		dZy =  self.df[j-1] * ey + self.df1Int[j-1] * ey1
		coeff =  -3. / (j  * 2 * (delta(self.pratio,j,j-1.))**2 )
		coeff2 = -1. / (j * delta(self.pratio,j,j-1.) )
		
		dl1x = -1. * coeff * Zy - coeff2 * (Zy + self.alpha * dZy)
		dl1y = -1. * coeff * Zx - coeff2 * (Zx + self.alpha * dZx)
	
		return array([dl1x,dl1y])
	
	
	def firstOrdCoeff(self,j,ex,ey,ex1,ey1):
		dlx,dly = self.dl(j,ex,ey,ex1,ey1)
		z = self.dz(j)
		vx,vy = z + dly, -dlx
	
		return array([vx,vy])
	
	def firstOrdCoeff1(self,j,ex,ey,ex1,ey1):
		dl1x,dl1y = self.dl1(j,ex,ey,ex1,ey1)
		z1 = self.dz1(j)
		v1x,v1y = z1 + dl1y, -dl1x
	
		return array([v1x,v1y])
	
	def get_FirstOrd_ttvfuncs(self,L0,L10,j,ex,ey,ex1,ey1):
		vx,vy = self.firstOrdCoeff(j,ex,ey,ex1,ey1)
		v1x,v1y = self.firstOrdCoeff1(j,ex,ey,ex1,ey1)
		cfn,sfn = sinusoids(self.p,self.p1,L0,L10,j,j-1)
		innerDt = lambda t: 1 * self.p * ( vx * sfn(t) + vy * cfn(t) ) / (pi)
		outerDt = lambda t: 1 * self.p1 * ( v1x * sfn(t) + v1y * cfn(t) ) / (pi)
	
		return array([innerDt,outerDt])
	
	def get_FirstOrd_ttvs(self,L0,L10,j,ex,ey,ex1,ey1):
		vx,vy = self.firstOrdCoeff(j,ex,ey,ex1,ey1)
		v1x,v1y = self.firstOrdCoeff1(j,ex,ey,ex1,ey1)
		cfn,sfn = sinusoids(self.p,self.p1,L0,L10,j,j-1)
		innerDt = array(1 * self.p * ( vx * sfn(self.transits) + vy * cfn(self.transits) ) / (pi))
		outerDt = array(1 * self.p1 * ( v1x * sfn(self.transits1) + v1y * cfn(self.transits1) ) / (pi))

		return innerDt,outerDt
	########################################################################
	# one-to-one terms
	########################################################################
	
	def dlO2O(self,j):
		term1 = 3. * self.pratio * self.kj[j-1] / ( sqrt(self.alpha) * (1. - self.pratio) )
		term2 = -2. * sqrt(self.alpha) * self.dkj[j-1]
	
		return array([ 0 , -1*(term1 + term2) / (j * (1-self.pratio) )])
	
	def dl1O2O(self,j):
		term1 = -3 * self.k1j[j-1]/(1-self.pratio) 
		term2 = 2 * self.alpha * self.dk1j[j-1]
		term3 = 2 * self.k1j[j-1]
	
		return array([ 0 , -1.* (term1 + term2 + term3) / (j * (1 - self.pratio) ) ])
	
	def one2oneCoeff(self,j):
		dlx,dly = self.dlO2O(j)
		vx,vy =  dly, -dlx
	
		return array([vx,vy])
	
	def one2oneCoeff1(self,j):
	
		dl1x,dl1y = self.dl1O2O(j)
		v1x,v1y =  dl1y, -dl1x
	
		return array([v1x,v1y])
	
	def get_One2One_ttvfuncs(self,L0,L10,j):
		vx,vy = self.one2oneCoeff(j)
		v1x,v1y = self.one2oneCoeff1(j)
		cfn,sfn = sinusoids(self.p,self.p1,L0,L10,j,j)
		innerDt = lambda t: 1 * self.p * ( vx * sfn(t) + vy * cfn(t) ) / (pi)
		outerDt = lambda t: 1 * self.p1 * ( v1x * sfn(t) + v1y * cfn(t) ) / (pi)
	
		return array([innerDt,outerDt])

	def get_One2One_ttvs(self,L0,L10,j):
		vx,vy = self.one2oneCoeff(j)
		v1x,v1y = self.one2oneCoeff1(j)
		cfn,sfn = sinusoids(self.p,self.p1,L0,L10,j,j)
		innerDt =  1 * self.p * ( vx * sfn(self.transits) + vy * cfn(self.transits) ) / (pi)
		outerDt =  1 * self.p1 * ( v1x * sfn(self.transits1) + v1y * cfn(self.transits1) ) / (pi)
	
		return innerDt,outerDt

	def get_All_One2One(self,L0,L10):
	
		inFnsO2O,outFnsO2O = array([get_One2One_ttvfuncs(L0,L10,j) for j in arange(1,6)]).T
		return (lambda t: sum( [fn(t) for fn in inFnsO2O ] ),\
			lambda t: sum( [fn(t) for fn in outFnsO2O] ))
		
	########################################################################
	# second-order terms
	########################################################################
	
	def dzScnd(self,j,ex,ey,ex1,ey1):
	
		Zx = 2*self.g[j-3] * ex + self.h[j-3] * ex1
		Zy = 2*self.g[j-3] * ey + self.h[j-3] * ey1
		
		coeff = -1. / ( sqrt(self.alpha) * j * delta(self.pratio,j,j-2) ) 
	
		return array([coeff*Zx, -1. * coeff*Zy])
	
	def dz1Scnd(self,j,ex,ey,ex1,ey1):
	
		Zx = 2*self.g1Int[j-3] * ex1 + self.h[j-3] * ex
		Zy = 2*self.g1Int[j-3] * ey1 + self.h[j-3] * ey
		
		coeff = -1. / ( j * delta(self.pratio,j,j-2) ) 
	
		return array([coeff*Zx, -1. * coeff*Zy])
	
	def dlScnd(self,j,ex,ey,ex1,ey1):
		Z2x = self.g[j-3] * (ex**2-ey**2) + self.g1Ext[j-3] * (ex1**2-ey1**2) + self.h[j-3] * (ex*ex1 - ey*ey1)
		Z2y = self.g[j-3] * (2*ex*ey) + self.g1Ext[j-3] * (2*ex1*ey1) + self.h[j-3] * ( ex1*ey + ey1*ex )
		coeff = 1.5 * self.pratio * (j-2) / (j * j *  sqrt(self.alpha) * delta(self.pratio,j,j-2)**2 )
		
		dlx = -1 * coeff * Z2y
		dly = -1 * coeff * Z2x
		
		return array([dlx,dly])
	 
	def dl1Scnd(self,j,ex,ey,ex1,ey1):
	
		Z2x = self.g[j-3] * (ex**2-ey**2) + self.g1Int[j-3] * (ex1**2-ey1**2) + self.h[j-3] * (ex*ex1 - ey*ey1)
		Z2y = self.g[j-3] * (2*ex*ey) + self.g1Int[j-3] * (2*ex1*ey1) + self.h[j-3] * ( ex1*ey + ey1*ex )
	
		coeff = -1.5 / (j * delta(self.pratio,j,j-2) **2 )
		
		dl1x = -1 * coeff * Z2y
		dl1y = -1 * coeff * Z2x
	
		return array([ dl1x,dl1y])
	
	def scndOrdCoeff(self,j,ex,ey,ex1,ey1):
		dlx,dly = self.dlScnd(j,ex,ey,ex1,ey1)
		zx,zy = self.dzScnd(j,ex,ey,ex1,ey1)
		vx,vy = zx + dly, zy-dlx
	
		return array([vx,vy])
	
	def scndOrdCoeff1(self,j,ex,ey,ex1,ey1):
		dl1x,dl1y = self.dl1Scnd(j,ex,ey,ex1,ey1)
		z1x,z1y = self.dz1Scnd(j,ex,ey,ex1,ey1)
		v1x,v1y = z1x + dl1y, z1y-dl1x
	
		return array([v1x,v1y])
	
	def get_ScndOrder_ttvfuncs(self,L0,L10,j,ex,ey,ex1,ey1):
		vx,vy = self.scndOrdCoeff(j,ex,ey,ex1,ey1)
		v1x,v1y = self.scndOrdCoeff1(j,ex,ey,ex1,ey1)
		cfn,sfn = sinusoids(self.p,self.p1,L0,L10,j,j-2)
		innerDt = lambda t: 1 * self.p * ( vx * sfn(t) + vy * cfn(t) ) / (pi)
		outerDt = lambda t: 1 * self.p1 * ( v1x * sfn(t) + v1y * cfn(t) ) / (pi)
	
		return array([innerDt,outerDt])

	def get_ScndOrder_ttvs(self,L0,L10,j,ex,ey,ex1,ey1):

		vx,vy = self.scndOrdCoeff(j,ex,ey,ex1,ey1)
		v1x,v1y = self.scndOrdCoeff1(j,ex,ey,ex1,ey1)
		cfn,sfn = sinusoids(self.p,self.p1,L0,L10,j,j-2)
		innerDt = 1 * self.p * ( vx * sfn(self.transits) + vy * cfn(self.transits) ) / (pi)
		outerDt = 1 * self.p1 * ( v1x * sfn(self.transits1) + v1y * cfn(self.transits1) ) / (pi)
	
		return innerDt,outerDt
	########################################################################
	# 
	########################################################################
	
	def totalFuncs(self,L0,L10,ex,ey,ex1,ey1,minj=1,maxj=5):
		
		inFns,outFns = array([ self.get_FirstOrd_ttvfuncs(L0,L10,j,ex,ey,ex1,ey1) for j in arange(0,6)]).T
		inFnsScnd,outFnsScnd = array([ self.get_ScndOrder_ttvfuncs(L0,L10,j,ex,ey,ex1,ey1) for j in arange(3,8)]).T
		inFnsO2O,outFnsO2O = array([self.get_One2One_ttvfuncs(L0,L10,j) for j in arange(minj,maxj+1)]).T
		return (lambda t: sum( [fn(t) for fn in append(append(inFns,inFnsO2O),inFnsScnd) ] ),\
			lambda t: sum( [fn(t) for fn in append(append(outFns,outFnsO2O),outFnsScnd) ] ))
	
	def totalTTVs(self,L0,L10,ex,ey,ex1,ey1,minj=1,maxj=5):
		
		inDts,outDts = map(sum,transpose([ self.get_FirstOrd_ttvs( L0, L10, j, ex, ey, ex1, ey1 ) for j in arange(0,6) ]))
		#inDtsScnd,outDtsScnd = map(sum,transpose([ self.get_ScndOrder_ttvs(L0,L10,j,ex,ey,ex1,ey1) for j in arange(3,8)]))
		inDtsScnd,outDtsScnd =  self.get_ScndOrder_ttvs(L0,L10,2*self.j,ex,ey,ex1,ey1) 
		inDtsO2O,outDtsO2O = map(sum,transpose([self.get_One2One_ttvs(L0,L10,j) for j in arange(minj,maxj+1)]))
		return inDts + inDtsScnd + inDtsO2O, outDts + outDtsScnd + outDtsO2O 	
	
	def selectTTVs(self,L0,L10,ex,ey,ex1,ey1,fo,so,oo):
		
		inDts,outDts = map(sum,transpose([ self.get_FirstOrd_ttvs( L0, L10, j, ex, ey, ex1, ey1 ) for j in fo ]))
		inDtsScnd,outDtsScnd = map(sum,transpose([ self.get_ScndOrder_ttvs(L0,L10,j,ex,ey,ex1,ey1) for j in so]))
		inDtsO2O,outDtsO2O = map(sum,transpose([self.get_One2One_ttvs(L0,L10,j) for j in oo ]))
		return inDts + inDtsScnd + inDtsO2O, outDts + outDtsScnd + outDtsO2O 	

	def get_ttvs(self,ex,ey,ex1,ey1,firstOrder=False):
		#
		transits = self.transits
		transits1 = self.transits
		th0 = 2*pi*(-1.*self.T0)/self.p
		#theta0 = th0 + 2 * ex * sin( th0 ) + 2 * ey *(1. - cos( th0 ) )
		th10 = 2*pi*(-1.*self.T10)/self.p1
		#theta10 = th10 + 2 * ex1 * sin( th10 ) + 2 * ey1 *( 1. -  cos( th10 ))
		L0 =  th0  + 2 * ( ey  ) 
		L10 = th10 + 2 * ( ey1 ) # 2.5 #
#		
		j = self.j
		if firstOrder:
			dt,dt1 = self.get_FirstOrd_ttvs(L0,L10,j,ex,ey,ex1,ey1)
		else:
			dt,dt1 = self.totalTTVs(L0,L10,ex,ey,ex1,ey1)
		
		return  dt , dt1 

	def select_ttvs(self,ex,ey,ex1,ey1,fo,so,oo):
		#
		transits = self.transits
		transits1 = self.transits
		th0 = 2*pi*(-1.*self.T0)/self.p
		th10 = 2*pi*(-1.*self.T10)/self.p1

		L0 =  th0  + 2 * ( ey  ) 
		L10 = th10 + 2 * ( ey1 ) # 2.5 #
#		
		dt,dt1 = self.selectTTVs(L0,L10,ex,ey,ex1,ey1,fo,so,oo)
		return  dt , dt1 
				
	def fitness(self,pars,firstOrder=False):
		
		ex,ey,ex1,ey1 = pars
	
		AnalyticTTVs = self.get_ttvs(ex,ey,ex1,ey1,firstOrder=firstOrder)
#		#
		pl0tr = self.transits
		pl1tr = self.transits1
		N = self.trN
		N1 = self.trN1
		errs,errs1 = self.input_data[:,2],self.input_data1[:,2]
#		#
		MassFitData = linalg.lstsq( array([time_to_ttv(N, AnalyticTTVs[0] )]).T , time_to_ttv(N,pl0tr) )
		MassFitData1 = linalg.lstsq( array([time_to_ttv(N1, AnalyticTTVs[1] )]).T , time_to_ttv(N1 , pl1tr) ) 
		mass,mass1 = MassFitData[0][0],MassFitData1[0][0]
#		#
		resids=time_to_ttv(N,pl0tr)  - array( time_to_ttv(N, AnalyticTTVs[0] ) * mass ) #- MassFitData[0][0] )
		resids1=time_to_ttv(N1,pl1tr)  - array( time_to_ttv(N1, AnalyticTTVs[1] ) * mass1 ) # - MassFitData1[0][0] )
#		#---------------------------------------------
		chi2=0.0
		chi2 = sum( resids**2 / errs**2 ) + sum( resids1**2/errs1**2 ) 
#	    	#---------------------------------------------
		return -1.0 * chi2, array([mass,mass1])
#	
##############################################################
#
	def fitness2(self,pars,firstOrder=False):
		"""Compute the fitness of analytic TTV model as a function of parameters 'pars'.
		parameters are taken to be 'pars' = [m,m1,ex,ey,ex1,ey1] """	
		m,m1,ex,ey,ex1,ey1 = pars
	
		AnalyticTTVs = self.get_ttvs(ex,ey,ex1,ey1,firstOrder=firstOrder)
#		#
		pl0tr = self.transits
		pl1tr = self.transits1
		N = self.trN
		N1 = self.trN1
		errs,errs1 = self.input_data[:,2],self.input_data1[:,2]
#		#
		#
		resids = pl0tr - self.p*N - self.T0  -  AnalyticTTVs[0] * m1  
		resids1 = pl1tr - self.p1*N1 - self.T10  -  AnalyticTTVs[1] * m  
#		#---------------------------------------------
		chi2=0.0
		chi2 = sum( resids**2 / errs**2 ) + sum( resids1**2/errs1**2 ) 
#	    	#---------------------------------------------
		return -1.0 * chi2 
#	
##############################################################
#	Debugging Functions
##############################################################
	def fitplot(self,pars):
		
		AnalyticTTVs = []
		for parset in pars:
			m,m1,ex,ey,ex1,ey1 = parset
			AnalyticTTVs.append( self.get_ttvs(ex,ey,ex1,ey1) )
#		#
		pl0tr = self.transits
		pl1tr = self.transits1
		N = self.trN
		N1 = self.trN1
		errs,errs1 = self.input_data[:,2],self.input_data1[:,2]
#		#
		symbols = ['x','o','d']
		## Figure 1 ##
		plt.figure()
		plt.subplot(211)
		plt.errorbar(pl0tr, pl0tr - self.p*N - self.T0,yerr=errs,fmt='ks')
		for i,ttvs in enumerate(AnalyticTTVs):
			plt.plot(pl0tr , ttvs[0] * m1 ,'k%s'% symbols[i%len(symbols)] ) 
		plt.subplot(212)
		plt.errorbar(pl1tr , pl1tr - self.p1*N1 - self.T10 ,yerr=errs1,fmt='rs')
		for i,ttvs in enumerate(AnalyticTTVs):
			plt.plot(pl1tr , ttvs[1] * m  ,'r%s'% symbols[i%len(symbols)] ) 
		plt.show()
#		#---------------------------------------------
	def residfitplot(self,pars):
		
		fo = array([k for k in arange(0,6) if k!= self.j])
		oo = arange(1,6)
		so = [2 * self.j]
		AnalyticTTVs = []
		for parset in pars:
			m,m1,ex,ey,ex1,ey1 = parset
			AnalyticTTVs.append( self.select_ttvs(ex,ey,ex1,ey1,fo,so,oo) )
#		#
		pl0tr = self.transits
		pl1tr = self.transits1
#		#
		symbols = ['x','o','d']
		## Figure 1 ##
		plt.figure()
		plt.subplot(211)
		fullPeriodFit2plot(self.input_data,self.input_data1,resids=True,planets=1)
		for i,ttvs in enumerate(AnalyticTTVs):
			plt.plot(pl0tr , ttvs[0] * m1 ,'k%s'% symbols[i%len(symbols)] ) 

		plt.subplot(212)
		fullPeriodFit2plot(self.input_data,self.input_data1,resids=True,planets=2)
		for i,ttvs in enumerate(AnalyticTTVs):
			plt.plot(pl1tr , ttvs[1] * m  ,'r%s'% symbols[i%len(symbols)] ) 
		plt.show()
#		#---------------------------------------------
	def selectplot(self,pars,fo,so,oo,planets=3):
		""" planets option: 1 - inner, 2 - outer, other - both """	
		m,m1,ex,ey,ex1,ey1 = pars
	
		AnalyticTTVs = self.select_ttvs(ex,ey,ex1,ey1,fo,so,oo)
#		#
		pl0tr = self.transits
		pl1tr = self.transits1
		N = self.trN
		N1 = self.trN1
#		#
		if planets ==1:
			plt.plot(pl0tr  ,  AnalyticTTVs[0] * m1 ,'k--') 
		elif planets==2:
			plt.plot(pl1tr ,   AnalyticTTVs[1] * m  ,'r--') 
		else:
			plt.plot(pl0tr  ,  AnalyticTTVs[0] * m1 ,'k--') 
			plt.plot(pl1tr ,   AnalyticTTVs[1] * m  ,'r--') 
		#plt.show()

#		#---------------------------------------------
	def ttvplot(self):
#		#
		pl0tr = self.transits
		pl1tr = self.transits1
		N = self.trN
		N1 = self.trN1
		plt.plot(pl0tr , pl0tr - self.p*N - self.T0)
		plt.plot(pl1tr , pl1tr - self.p1*N1 - self.T10 )
		plt.show()
		return
#		#---------------------------------------------
#	
	def SummaryTable(self,pars):
		m,m1,ex,ey,ex1,ey1 = pars
		th0 = 2*pi*(-1.*self.T0)/self.p
		th10 = 2*pi*(-1.*self.T10)/self.p1
		L0 =  th0  + 2 * ( ey  ) 
		L10 = th10 + 2 * ( ey1 ) 
##
##
		print "P: %.6f \t P1: %.6f " % (self.p,self.p1)
		print "L0: %.2f \t L10: %.2f " % (L0,L10)
		print 
		print "Inner Planet"
		print "res.\tdz\tdl"
		for i in range(6):
			lx,ly = self.dl(i,ex,ey,ex1,ey1)
			print "%d / %d \t %.3f \t %.3f + i*%.3f" % (i,i-1,self.dz(i) ,lx,ly )
		for i in range(3,8):
			lx,ly = self.dlScnd(i,ex,ey,ex1,ey1)
			zx,zy = self.dzScnd(i,ex,ey,ex1,ey1)
			print "%d / %d \t %.3f+i*%.3f \t %.3f + i*%.3f" % (i,i-2,zx,zy,lx,ly )
		for i in range(1,6):
			lx,ly = self.dlO2O(i)
			print "%d / %d \t %.3f  \t %.3f + i*%.3f" % (i,i,0.,lx,ly)
		print "Outer Planet"
		print "res.\tdz\tdl"
		for i in range(1,6):
			lx,ly = self.dl1(i,ex,ey,ex1,ey1)
			print "%d / %d \t %.3f \t %.3f + i*%.3f" % (i,i-1,self.dz1(i) ,lx,ly )
		for i in range(3,8):
			lx,ly = self.dl1Scnd(i,ex,ey,ex1,ey1)
			zx,zy = self.dz1Scnd(i,ex,ey,ex1,ey1)
			print "%d / %d \t %.3f+i*%.3f \t %.3f + i*%.3f" % (i,i-2,zx,zy,lx,ly )
		for i in range(1,6):
			lx,ly = self.dl1O2O(i)
			print "%d / %d \t %.3f  \t %.3f + i*%.3f" % (i,i,0.,lx,ly)

		
			

	########################################################################################
	########################################################################################
	########################################################################################
if __name__=="__main__":
	import timeit
	input_data=loadtxt("./inner.ttv")
	input_data1 = loadtxt("./outer.ttv")
	ft = fitness(input_data,input_data1)
	evals = random.normal(0,0.1,size=4)
	s="""\
from numpy import loadtxt
from numpy import random
from __main__ import fitness
input_data=loadtxt("./inner.ttv")
input_data1 = loadtxt("./outer.ttv")
ft = fitness(input_data,input_data1)
evals = random.normal(0,0.1,size=4)"""
	if False:
		exectimeFO = timeit.timeit(stmt='ft.fitness(evals,firstOrder=True)',setup=s,number=100)
		print "100 executions of fitness,first-order only, in %g seconds" % exectimeFO
		exectime = timeit.timeit('ft.fitness(evals)',setup=s,number=100)
		print "100 executions of fitness in %g seconds" % exectime
	
	########################################################################################
	########################################################################################
	########################################################################################
