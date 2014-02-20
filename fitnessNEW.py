from numpy import *
import LaplaceCoefficients as LC
#---------------------------------------------------
# miscellaneous
#---------------------------------------------------
def linearfit(x,y):
      assert len(x)==len(y), "length of arrays unequal! x: {}, y: {}".format(len(x),len(y))
      a = array([ones(len(x)),x]).T
      return linalg.lstsq(a,y)[0]

def delta(pratio,j,k):
	return pratio * k/j - 1.
		
def get_res(pratio):
	return argmin([abs(delta(pratio,j,j-1)) for j in range(2,6)]) + 2
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
	#---------------------------------------------------
	# fill laplace coefficient arrays
	#---------------------------------------------------
	def __init__(self,p,p1,input_data,input_data1):
		self.p = p
		self.p1 = p1
		self.pratio = p1/p
		self.self.alpha = (self.pratio)**(-2./3.)
		self.f = get_f_array(pratio)
		self.f1Ext = get_f1Ext_array(pratio)
		self.f1Int = get_f1Int_array(pratio)
		self.k = get_k_array(pratio)
		self.k1 = get_k1_array(pratio)
		self.g = get_g_array(pratio)
		self.g1Ext = get_g1Ext_array(pratio)
		self.g1Int = get_g1Int_array(pratio)
		self.h = get_h_array(pratio)
		self.input_data = input_data
		self.input_data1 = input_data1	
	########################################################################
	########################################################################
	# First-order terms
	########################################################################
	def dz(self,j):
		return -1.0 * f[j-2] / ( sqrt(self.alpha)  * j * delta(self.pratio,j,j-1)  )
		
	def dz1(self,j):
		return -1.0  * f1Int[j-2] / (j * delta(self.pratio,j,j-1))
	
	def dl(self,j,ex,ey,ex1,ey1):
		Zx = f[j-2] * ex + f1Ext[j-2] * ex1
		Zy = f[j-2] * ey + f1Ext[j-2] * ey1
		dZx = df[j-2] * ex + df1Ext[j-2] * ex1
		dZy =  df[j-2] * ey + df1Ext[j-2] * ey1
		coeff = (j-1.) * 3. / (j * 2 * self.alpha**2 * j *  (delta(self.pratio,j,j-1.))**2 )
		coeff2 = sqrt(self.alpha) / (j * delta(self.pratio,j,j-1.))
		# This is correct-- note 1/i factor and Zconj in definition	
		dlx = -1. * coeff * Zy - coeff2 * dZy
		dly = -1. * coeff * Zx - coeff2 * dZx
	
		return (dlx,dly)
	
		
	def dl1(self,j,ex,ey,ex1,ey1):
		Zx = f[j-2] * ex + f1Int[j-2] * ex1
		Zy = f[j-2] * ey + f1Int[j-2] * ey1
		dZx = df[j-2] * ex + df1Int[j-2] * ex1
		dZy =  df[j-2] * ey + df1Int[j-2] * ey1
		coeff =  -3. / (j  * 2 * (delta(self.pratio,j,j-1.))**2 )
		coeff2 = -1. / (j * delta(self.pratio,j,j-1.) )
		
		dl1x = -1. * coeff * Zy - coeff2 * (Zy + self.alpha * dZy)
		dl1y = -1. * coeff * Zx - coeff2 * (Zx + self.alpha * dZx)
	
		return (dl1x,dl1y)
	
	
	def firstOrdCoeff(self,j,ex,ey,ex1,ey1):
		dlx,dly = self.dl(j,ex,ey,ex1,ey1)
		z = self.dz(j)
		vx,vy = z + dly, -dlx
	
		return (vx,vy)
	
	def firstOrdCoeff1(self,j,ex,ey,ex1,ey1):
		dl1x,dl1y = self.dl1(j,ex,ey,ex1,ey1)
		z1 = self.dz1(j)
		v1x,v1y = z1 + dl1y, -dl1x
	
		return (v1x,v1y)
	
	def get_FirstOrd_ttvfuncs(self,p,p1,L0,L10,j,ex,ey,ex1,ey1):
		self.pratio = p1/p
		vx,vy = self.firstOrdCoeff(self.pratio,j,ex,ey,ex1,ey1)
		v1x,v1y = self.firstOrdCoeff1(self.pratio,j,ex,ey,ex1,ey1)
		cfn,sfn = sinusoids(p,p1,L0,L10,j,j-1)
		innerDt = lambda t: 1 * p * ( vx * sfn(t) + vy * cfn(t) ) / (pi)
		outerDt = lambda t: 1 * p1 * ( v1x * sfn(t) + v1y * cfn(t) ) / (pi)
	
		return (innerDt,outerDt)
	
	########################################################################
	# one-to-one terms
	########################################################################
	
	def dlO2O(self.pratio,j):
		term1 = 3. * self.pratio * kj[j-1] / ( sqrt(self.alpha) * (1. - self.pratio) )
		term2 = -2. * sqrt(self.alpha) * dkj[j-1]
	
		return ( 0 , -1*(term1 + term2) / (j * (1-self.pratio) ))
	
	def dl1O2O(self.pratio,j):
		term1 = -3 * k1j[j-1]/(1-self.pratio) 
		term2 = 2 * self.alpha * dk1j[j-1]
		term3 = 2 * k1j[j-1]
	
		return ( 0 , -1.* (term1 + term2 + term3) / (j * (1 - self.pratio) ) )
	
	def one2oneCoeff(self.pratio,j):
		dlx,dly = dlO2O(self.pratio,j)
		vx,vy =  dly, -dlx
	
		return (vx,vy)
	
	def one2oneCoeff1(self.pratio,j):
	
		dl1x,dl1y = dl1O2O(self.pratio,j)
		v1x,v1y =  dl1y, -dl1x
	
		return (v1x,v1y)
	
	def get_One2One_ttvfuncs(p,p1,L0,L10,j):
		self.pratio = p1/p
		vx,vy = one2oneCoeff(self.pratio,j)
		v1x,v1y = one2oneCoeff1(self.pratio,j)
		cfn,sfn = sinusoids(p,p1,L0,L10,j,j)
		innerDt = lambda t: 1 * p * ( vx * sfn(t) + vy * cfn(t) ) / (pi)
		outerDt = lambda t: 1 * p1 * ( v1x * sfn(t) + v1y * cfn(t) ) / (pi)
	
		return (innerDt,outerDt)
	
	def get_All_One2One(p,p1,L0,L10):
	
		inFnsO2O,outFnsO2O = array([get_One2One_ttvfuncs(p,p1,L0,L10,j) for j in arange(1,6)]).T
		return (lambda t: sum( [fn(t) for fn in inFnsO2O ] ),\
			lambda t: sum( [fn(t) for fn in outFnsO2O] ))
		
	########################################################################
	# second-order terms
	########################################################################
	
	def dzScnd(self.pratio,j,ex,ey,ex1,ey1):
	
		Zx = 2*g[j-3] * ex + h[j-3] * ex1
		Zy = 2*g[j-3] * ey + h[j-3] * ey1
		
		coeff = -1. / ( sqrt(self.alpha) * j * delta(self.pratio,j,j-2) ) 
	
		return (coeff*Zx, -1. * coeff*Zy)
	
	def dz1Scnd(self.pratio,j,ex,ey,ex1,ey1):
	
		Zx = 2*g1Int[j-3] * ex1 + h[j-3] * ex
		Zy = 2*g1Int[j-3] * ey1 + h[j-3] * ey
		
		coeff = -1. / ( j * delta(self.pratio,j,j-2) ) 
	
		return (coeff*Zx, -1. * coeff*Zy)
	
	def dlScnd(self.pratio,j,ex,ey,ex1,ey1):
		Z2x = g[j-3] * (ex**2-ey**2) + g1Ext[j-3] * (ex1**2-ey1**2) + h[j-3] * (ex*ex1 - ey*ey1)
		Z2y = g[j-3] * (2*ex*ey) + g1Ext[j-3] * (2*ex1*ey1) + h[j-3] * ( ex1*ey + ey1*ex )
		coeff = 1.5 * self.pratio * (j-2) / (j * j *  sqrt(self.alpha) * delta(self.pratio,j,j-2)**2 )
		
		dlx = -1 * coeff * Z2y
		dly = -1 * coeff * Z2x
		
		return(dlx,dly)
	 
	def dl1Scnd(self.pratio,j,ex,ey,ex1,ey1):
	
		Z2x = g[j-3] * (ex**2-ey**2) + g1Int[j-3] * (ex1**2-ey1**2) + h[j-3] * (ex*ex1 - ey*ey1)
		Z2y = g[j-3] * (2*ex*ey) + g1Int[j-3] * (2*ex1*ey1) + h[j-3] * ( ex1*ey + ey1*ex )
	
		coeff = -1.5 / (j * delta(self.pratio,j,j-2) **2 )
		
		dl1x = -1 * coeff * Z2y
		dl1y = -1 * coeff * Z2x
	
		return( dl1x,dl1y)
	
	def scndOrdCoeff(self.pratio,j,ex,ey,ex1,ey1):
		dlx,dly = dlScnd(self.pratio,j,ex,ey,ex1,ey1)
		zx,zy = dzScnd(self.pratio,j,ex,ey,ex1,ey1)
		vx,vy = zx + dly, zy-dlx
	
		return (vx,vy)
	
	def scndOrdCoeff1(self.pratio,j,ex,ey,ex1,ey1):
		dl1x,dl1y = dl1Scnd(self.pratio,j,ex,ey,ex1,ey1)
		z1x,z1y = dz1Scnd(self.pratio,j,ex,ey,ex1,ey1)
		v1x,v1y = z1x + dl1y, z1y-dl1x
	
		return (v1x,v1y)
	
	def get_ScndOrder_ttvfuncs(p,p1,L0,L10,j,ex,ey,ex1,ey1):
		self.pratio = p1/p
		vx,vy = scndOrdCoeff(self.pratio,j,ex,ey,ex1,ey1)
		v1x,v1y = scndOrdCoeff1(self.pratio,j,ex,ey,ex1,ey1)
		cfn,sfn = sinusoids(p,p1,L0,L10,j,j-2)
		innerDt = lambda t: 1 * p * ( vx * sfn(t) + vy * cfn(t) ) / (pi)
		outerDt = lambda t: 1 * p1 * ( v1x * sfn(t) + v1y * cfn(t) ) / (pi)
	
		return (innerDt,outerDt)
	########################################################################
	# 
	########################################################################
	
	def totalFuncs(p,p1,L0,L10,ex,ey,ex1,ey1,minj=1,maxj=5):
		inFns,outFns = array([ get_FirstOrd_ttvfuncs(p,p1,L0,L10,j,ex,ey,ex1,ey1) for j in arange(2,6)]).T
		inFnsScnd,outFnsScnd = array([ get_ScndOrder_ttvfuncs(p,p1,L0,L10,j,ex,ey,ex1,ey1) for j in arange(3,8)]).T
		inFnsO2O,outFnsO2O = array([get_One2One_ttvfuncs(p,p1,L0,L10,j) for j in arange(minj,maxj+1)]).T
		return (lambda t: sum( [fn(t) for fn in append(append(inFns,inFnsO2O),inFnsScnd) ] ),\
			lambda t: sum( [fn(t) for fn in append(append(outFns,outFnsO2O),outFnsScnd) ] ))
	
	def sinusoids(p,p1,L0,L10,j,k):
	
		freq = 2. * pi * ( ( j / p1) - ( k / p) ) 
		phi0 = j * L10  - k * L0
		sinfn =  lambda t: sin(freq * t + phi0)
		cosfn =  lambda t: cos(freq * t + phi0)
		
		return (cosfn,sinfn)
	
	
	
	def get_ttvs(input_data,input_data1,ex,ey,ex1,ey1,firstOrder=False):
		#
		transits = input_data[:,1]
		transits1 = input_data1[:,1]
		#
		T0,p = linearfit(input_data[:,0],input_data[:,1])
		T10,p1 = linearfit(input_data1[:,0],input_data1[:,1])
		th0 = 2*pi*(-1.*T0)/p
		theta0 = th0 + 2 * ex * sin( th0 ) + 2 * ey *(1. - cos( th0 ) )
		th10 = 2*pi*(-1.*T10)/p1
		theta10 = th10 + 2 * ex1 * sin( th10 ) + 2 * ey1 *( 1. -  cos( th10 ))
		L0 =  theta0  + 2 * ( ey * cos(theta0) - ex * sin(theta0) ) 
		L10 = theta10 + 2 *( ey1 * cos(theta10) - ex1 * sin(theta10) ) # 2.5 #
		#
		j = get_res(p1/p)
		if firstOrder:
			dt,dt1 = get_FirstOrd_ttvfuncs(p,p1,L0,L10,j,ex,ey,ex1,ey1)
		else:
			dt,dt1 = totalFuncs(p,p1,L0,L10,ex,ey,ex1,ey1)
		
		return ( array(map(dt,transits)) , array(map(dt1,transits1)) )
				
	def fitness(pars,input_data,input_data1,firstOrder=False):
		
		ex,ey,ex1,ey1 = pars
	
		AnalyticTTVs = get_ttvs(input_data[:,0:2],input_data1[:,0:2],ex,ey,ex1,ey1,firstOrder)
		#
		pl0tr = input_data[:,1] # Transit times of inner and outer planet
		pl1tr = input_data1[:,1]
		errs,errs1 = input_data[:,2],input_data1[:,2]
		#
		MassFitData = linalg.lstsq( array([ones(len(AnalyticTTVs[0])),time_to_ttv( AnalyticTTVs[0] )]).T , time_to_ttv(pl0tr) )
		MassFitData1 = linalg.lstsq( array([ones(len(AnalyticTTVs[1])),time_to_ttv( AnalyticTTVs[1] )]).T , time_to_ttv( pl1tr ) ) 
		mass,mass1 = MassFitData[0][1],MassFitData1[0][1]
		#
		resids=time_to_ttv(pl0tr)  - array( time_to_ttv( AnalyticTTVs[0] ) * MassFitData[0][1] - MassFitData[0][0] )
		resids1=time_to_ttv(pl1tr)  - array( time_to_ttv( AnalyticTTVs[1] ) * MassFitData1[0][1] - MassFitData1[0][0] )
		#---------------------------------------------
		chi2=0.0
		chi2 = sum( resids**2 / errs**2 ) + sum( resids1**2/errs1**2 )
	    	#---------------------------------------------
		if ex**2 + ey**2 < 1. and ex1**2 + ey1**2 < 1.:
			return -1.0 * chi2, array([mass,mass1])
		else:
			return -inf,array([mass,mass1])
	
	########################################################################################
	########################################################################################
	########################################################################################
	def testPlot(p,p1,L0,L10,j,ex,ey,ex1,ey1,terms=1):
		if terms == 1:
			inr,otr =  get_FirstOrd_ttvfuncs(p,p1,L0,L10,j,ex,ey,ex1,ey1)
		elif terms == 2:
			inr,otr = get_All_One2One(p,p1,L0,L10)
		elif terms == 3:
			inr,otr = get_ScndOrder_ttvfuncs(p,p1,L0,L10,j,ex,ey,ex1,ey1)
		t = linspace(0,100,400)
		plot(t,map(inr,t))
		plot(t,map(otr,t))
		savetxt("/tmp/inr.txt",array([t,map(inr,t)]).T)
		savetxt("/tmp/otr.txt",array([t,map(otr,t)]).T)
		show()
	def time_to_ttv(time):
		n = arange(len(time))
		a = array([ones(len(time)),n]).T
		b,m=linalg.lstsq(a, time)[0]
		return time-m*n-b
	########################################################################################
	########################################################################################
	########################################################################################
if __name__=="__main__":
	pl0tr=array([[i,t] for i,t in enumerate(loadtxt("/Users/samuelhadden/TTVOutputs/newdir/planet0.ttv").T[0])])
	pl1tr=array([[i,t] for i,t in enumerate(loadtxt("/Users/samuelhadden/TTVOutputs/newdir/planet1.ttv").T[0])])
	T0,p = linearfit(pl0tr[:,0],pl0tr[:,1])
	T10,p1 = linearfit(pl1tr[:,0],pl1tr[:,1])

	inputInner=hstack([ pl0tr , 0.0003 *  ones((len(pl0tr),1)) ])
	inputOuter=hstack([ pl1tr , 0.0003 *  ones((len(pl1tr),1)) ])

	mypars = [0.00760417*cos(pi*356.426/180),00.00760417*sin(pi*356.426/180),0.028*cos(1.563),0.028*sin(1.563)]
	print fitness(mypars,inputInner,inputOuter)
