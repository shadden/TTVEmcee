import sys
sys.path.insert(0,'/Users/samuelhadden/13_HighOrderTTV/TTVEmcee')
import numpy as np
from itertools import combinations
import matplotlib.pyplot as pl
import LaplaceCoefficients as LC
from scipy.optimize import curve_fit,leastsq



def linefit(x,y,sigma=None):
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	linefn = lambda x,slope,intercept: x*slope + intercept
	return curve_fit(linefn,x,y,sigma=sigma)
def linefit_resids(x,y,sigma=None):
	s,m = linefit(x,y,sigma)[0]
	return y - s*x -m
def delta(pratio,j,k):
	return pratio * k/j - 1.
def get_res(pratio):
	return np.argmin([np.abs(delta(pratio,j,j-1)) for j in range(2,6)]) + 2
	
class MultiplanetAnalyticTTVSystem(object):
	def __init__(self,inputData,deltaLimit=0.06):
		""" :deltaLimit float: limiting absolute value of delta for which to include 2S TTV component"""
		# Store observed transit information
		self.nPlanets = len(inputData)
		self.observedData= inputData
		self.transitNumbers = [ (data[:,0]).astype(int) for data in self.observedData] 
		self.transitCounts = [len(dat) for dat in self.observedData]
		self.transitTimes = [ data[:,1] for data in self.observedData ]
		self.transitUncertainties = [ data[:,2] for data in self.observedData ] 
		self.deltaLimit = deltaLimit
		# Flattened list of transit times and uncertainties, used for fitting the time scale/epoch
		self.flatTimes = np.hstack([times for times in self.transitTimes])
		self.flatUncertainties = np.hstack([unc for unc in self.transitUncertainties]) 
		
		
		# Estimate periods and T0's
		self.periodEstimates,self.tInitEstimates = np.array([linefit(data[:,0],data[:,1])[0] for data in inputData ]).T
		
		# Identify which planet pairs are near first-order resonance and store Laplace coefficients for pairs
		self.resonanceData = {}
		for pair in combinations(range(self.nPlanets),2):
			index1,index2 = pair
			p1,p2 = self.periodEstimates[index1], self.periodEstimates[index2]
			assert p1<p2, "Planets out of order!"
			pRatio = p2/p1
			j = get_res(pRatio)
			pairDelta = delta(pRatio,j,j-1.)
			alpha = np.power( pRatio, -2./3.)

			
			
			# Get the Laplace coefficients for this pair
				#-- 1 S/F --#
					#- 0:-1 terms -#
			f0 = -0.5 * alpha * LC.b(alpha,0.5,0,1)
			f10 = 1.5 / alpha**2 - 0.5 * LC.b(alpha,0.5,1,0) + 0.5 * alpha * LC.b(alpha,0.5,1,1)
			df0 = 0.0
			df10 = 0.0
			#
			laplaceCoeffF1ext = np.append( np.array((f10)) , LC.get_f1Ext_array(pRatio) )
			laplaceCoeffF1int = np.append( np.array((f10)) , LC.get_f1Int_array(pRatio) )
			laplaceCoeffF     = np.append( np.array((f0 )) , LC.get_f_array(pRatio)     )
			#
			laplaceCoeffdF1ext = np.append( np.array((df10)) , LC.get_df1Ext_array(pRatio) )
			laplaceCoeffdF1int = np.append( np.array((df10)) , LC.get_df1Int_array(pRatio) )
			laplaceCoeffdF =  np.append( np.array((df0 )) , LC.get_df_array(pRatio)     )
				# -- 0F --#
			laplaceCoeffK = LC.get_k_array(pRatio)
			laplaceCoeffK1 = LC.get_k1_array(pRatio)
			laplaceCoeffdK = LC.get_dk_array(pRatio)
			laplaceCoeffdK1 = LC.get_dk1_array(pRatio)
				#-- 2S --#
			laplaceCoeffG = LC.get_g_array(pRatio)
			laplaceCoeffG1 = LC.get_g1Int_array(pRatio)
			laplaceCoeffH = LC.get_h_array(pRatio)
			
							
			
			
			resDataDict = {'j':j,'Delta':pairDelta,\
				'f':laplaceCoeffF,'f1e':laplaceCoeffF1ext,'f1i':laplaceCoeffF1int,\
				'df':laplaceCoeffdF,'df1e':laplaceCoeffdF1ext,'df1i':laplaceCoeffF1int,\
				'k':laplaceCoeffK,'k1':laplaceCoeffK1,'dk':laplaceCoeffdK,'dk1':laplaceCoeffdK1,\
				'g':laplaceCoeffG,'g1':laplaceCoeffG1,'h':laplaceCoeffH}
			
			self.resonanceData.update({pair:resDataDict})
		
	def complexTTVAmplitudes1FS(self,massesAndEccs,periodRatio,resData):
		"""
		Get the first-order mean-motion resonance complex TTV amplitudes for a pair of planets.
		"""

		# Input variable parameters
		mu,mu1 = massesAndEccs[:2]
		ex,ey,ex1,ey1 = massesAndEccs[2:]
		
		# Input resonace information for pair
		j = np.arange(1,6)
		delta = (j-1.)*periodRatio /j -1.
		alpha = np.power( periodRatio, -2./3.)
		f,f1e,f1i,df,df1e,df1i = map(lambda x: resData[x],('f','f1e','f1i','df','df1e','df1i'))
		# Set up Laplace coefficients to correspond to to the array j=[1,2,3,4,5]
		f0,f10,df0,df10 = f[0],f1e[0],df[0],df1e[0]
		f,f1e,f1i,df,df1e,df1i = f[1:],f1e[1:],f1i[1:],df[1:],df1e[1:],df1i[1:]
		
		# Add indirect term to the 1:0 Laplace coefficient for external perturber
		fe,dfe = f.copy(),df.copy()
		#fe[0] += 1.5*alpha
		#dfe[0] += 1.5
		
		# Complex eccentricities
		Zxe,Zxi = fe * ex + f1e * ex1, f * ex + f1i * ex1
		Zye,Zyi = fe * ey + f1e * ey1,  f * ey + f1i * ey1

		Vx= mu1/np.sqrt(alpha) * (-fe/(j * delta) -(j-1.)/j * 1.5/(j*alpha**1.5 * delta**2 ) * Zxe)
		Vy= mu1/np.sqrt(alpha) *  ( -(j-1.)/j * 1.5/(j*alpha**1.5 * delta**2 )) * -Zye
		#-- extra lambda terms --#
		Vx+= -mu1*sqrt(alpha)/(j*delta) *  ((dfe - 0.25*f/sqrt(alpha)) * ex + df1e * ex1 )
		Vy+= -mu1*sqrt(alpha)/(j*delta) *  ((dfe - 0.25*f/sqrt(alpha)) * -ey + df1e * -ey1 )
		
		
		V1x = mu*(-f1i/(j*delta) + 1.5* Zxi/(j*delta**2) )
		V1y = mu*(1.5 * -Zyi/(j*delta**2) )
		#-- extra lambda terms --#
		V1x+= mu/(j*delta) *  ((f+alpha*df) * ex + (f1i+alpha*df1i - 0.25*f1i) * ex1 )
		V1y+= mu/(j*delta) *  ((f+alpha*df) * -ey + (f1i+alpha*df1i - 0.25*f1i) * -ey1 )
		
		#---------------------------------------------------#
		# prepend 0:-1 values 
		j = np.append( 0 , j)
		Vx0 = mu1 * f0 / ( np.sqrt(alpha) *periodRatio )
		V1x0 = mu * f10 / ( periodRatio )
		
		Vx = np.append( Vx0 , Vx)
		Vy = np.append( 0.0 , Vy)
		V1x = np.append( V1x0 , V1x)
		V1y = np.append( 0.0 , V1y)
		
		#
		return np.vstack((j,Vx,Vy,V1x,V1y)).T

	def complexTTVAmplitudes0F(self,massesAndEccs,pRatio,resData):
		"""
		Get the first-order mean-motion resonance complex TTV amplitudes for a pair of planets.
		"""

		# Input variable parameters
		mu,mu1 = massesAndEccs[:2]
		ex,ey,ex1,ey1 = massesAndEccs[2:]
		alpha = np.power( pRatio, -2./3.)

		
		# Input resonace information for pair
		j = np.arange(1,6)
		k,k1,dk,dk1 = map(lambda x: resData[x],('k','k1','dk','dk1'))
		
		Vx = -mu1 / (j*(1-pRatio)) * ( 3.*pRatio * k/ ( np.sqrt(alpha) * (1.-pRatio) ) -2.*np.sqrt(alpha)*dk )
		V1x = mu / (j*(1-pRatio))  * ( 3. * k1/(1.-pRatio) -  2 * (k1+alpha*dk1)  )
		
		return np.vstack((j,Vx,V1x)).T
	
	def complexTTVAmplitudes2S(self,massesAndEccs,pRatio,resData):
		#
		mu,mu1 = massesAndEccs[:2]
		ex,ey,ex1,ey1 = massesAndEccs[2:]
		alpha = np.power( pRatio, -2./3.)
		#
		j = 2 * resData['j']
		g,g1,h = map(lambda x: resData[x][j-3],('g','g1','h'))
		delta = (j-2) * pRatio/j - 1.
		#--------------------------------------------
		# -- dz terms -- #
		dzx = -mu1/(j * np.sqrt(alpha)* delta) * (2*g*ex + h * ex1)
		dzy = -mu1/(j * np.sqrt(alpha)* delta) * (2*g* -ey + h * -ey1)
		dz1x= -mu / (j*delta) * (2 * g1 * ex1 + h * ex )
		dz1y= -mu / (j*delta) * (2 * g1 * -ey1 + h * -ey )
		#
		# -- dl terms -- #
		bigZ2x = g * (ex*ex -ey*ey) + g1 * (ex1*ex1 - ey1*ey1) + h * (ex*ex1 - ey*ey1)
		bigZ2y = g * (2*ex*ey) + g1 * (2*ex1*ey1) + h * (ex*ey1 + ey*ex1)
		#
		dlx = -mu1 * 1.5/(j*delta**2) *(j-2.)/(j*alpha**2) * bigZ2x
		dly = -mu1 * 1.5/(j*delta**2) *(j-2.)/(j*alpha**2) * -bigZ2y
		dl1x = mu * 1.5/(j*delta**2) * bigZ2x
		dl1y = mu * 1.5/(j*delta**2) * -bigZ2y
		#--------------------------------------------		
		return np.array((j,dzx + dlx, dzy + dly, dz1x + dl1x, dz1y + dl1y))

	def TransitTimes(self,massesAndEccs,periodsAndMeanLongs):
		"""
		Compute transit times for each planet in the multi-planet system as the sum of a linear
		trend plus the TTVs induced by each partner.
		
		Parameters
		----------
		massesAndEccs : array 
			A list of the masses and two eccentricity vector components for consecutive planets.
			This can be a flat array or an array of triplets.
		periodsAndMeanLongs : array
			The average periods and initial mean longitudes of consecutive planets.  Mean longitudes
			should be given in RADIANS and are measured relative to the line of sight to the observer.
			This can be a flat array or an array of pairs.
		"""
		mass,ex,ey = np.transpose(massesAndEccs.reshape(-1,3))
		period,meanLong = np.transpose(periodsAndMeanLongs.reshape(-1,2))

		tInit = 0.5 * period *(-meanLong +2*ey) / np.pi #+ 0.5 * period * ( 2. * ey / np.pi )
		
		transitTimes = [ (period[i] * self.transitNumbers[i] + tInit[i]) for i in range(self.nPlanets) ]
		TTVs  = [ np.zeros(self.transitCounts[i]) for i in range(self.nPlanets) ]

		for pair,data in self.resonanceData.iteritems():
			i,j = pair
			
			parameters = np.array(( mass[i],mass[j],ex[i],ey[i],ex[j],ey[j] ))
			periodRatio = period[j] / period[i]
			
			ttv1FSdata = self.complexTTVAmplitudes1FS(parameters,periodRatio,data)
			ttv0Fdata = self.complexTTVAmplitudes0F(parameters,periodRatio,data)
			
			# Sum TTV contributions of 1F/1S terms
			for entry in ttv1FSdata:
				jRes,Vx,Vy,V1x,V1y = entry

				omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes-1)/period[i] )
				angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - (jRes - 1) * meanLong[i]
				angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - (jRes - 1) * meanLong[i]

				TTVs[i] += period[i] / np.pi * ( Vx * np.sin(angleResInner) + Vy * np.cos(angleResInner) )
				TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter) + V1y * np.cos(angleResOuter) )
			
			# Sum TTV contributions of 0F terms
			for entry in ttv0Fdata:
				jRes,Vx,V1x = entry

				omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes)/period[i] )
				angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - jRes * meanLong[i]
				angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - jRes * meanLong[i]

				TTVs[i] += period[i] / np.pi * ( Vx * np.sin(angleResInner) )
				TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter))

			# Add contribution of 2S term if delta is small
			if np.abs(data['Delta']) < self.deltaLimit:
				jRes,Vx,Vy,V1x,V1y = self.complexTTVAmplitudes2S(parameters,periodRatio,data)
				omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes-2)/period[i] )
				angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - (jRes - 2) * meanLong[i]
				angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - (jRes - 2) * meanLong[i]
				TTVs[i] += period[i] / np.pi * ( Vx  * np.sin(angleResInner) + Vy  * np.cos(angleResInner) )
				TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter) + V1y * np.cos(angleResOuter) )

				
		
		unscaledTimes =  [ time + dt  for time,dt in zip(transitTimes,TTVs) ]
		
		return [ np.vstack((self.transitNumbers[i],times)).T for i,times in enumerate(unscaledTimes) ]
	
	def TransitTimes_1SOnly(self,massesAndEccs,periodsAndMeanLongs):
		"""
		Compute transit times for each planet in the multi-planet system as the sum of a linear
		trend plus the TTVs induced by partners near first-order mean-motion resonance.
		
		Parameters
		----------
		massesAndEccs : array 
			A list of the masses and two eccentricity vector components for consecutive planets.
			This can be a flat array or an array of triplets.
		periodsAndMeanLongs : array
			The average periods and initial mean longitudes of consecutive planets.  Mean longitudes
			should be given in RADIANS and are measured relative to the line of sight to the observer.
			This can be a flat array or an array of pairs.
		"""
		mass,ex,ey = np.transpose(massesAndEccs.reshape(-1,3))
		period,meanLong = np.transpose(periodsAndMeanLongs.reshape(-1,2))

		tInit = 0.5 * period *(-meanLong +2*ey) / np.pi #+ 0.5 * period * ( 2. * ey / np.pi )
		
		transitTimes = [ (period[i] * self.transitNumbers[i] + tInit[i]) for i in range(self.nPlanets) ]
		TTVs  = [ np.zeros(self.transitCounts[i]) for i in range(self.nPlanets) ]

		for pair,data in self.resonanceData.iteritems():
			i,j = pair
			if np.abs(data['Delta']) > self.deltaLimit:
				continue
			
			parameters = np.array(( mass[i],mass[j],ex[i],ey[i],ex[j],ey[j] ))
			periodRatio = period[j] / period[i]
			jRes = data['j']
			ttv1FSdata = self.complexTTVAmplitudes1FS(parameters,periodRatio,data)
			# Get 1S terms and add their TTV contributions
			jvals,Vx,Vy,V1x,V1y = ttv1FSdata.T
			indx  = jvals.tolist().index(jRes)
			Vx,Vy,V1x,V1y = map(lambda x: x[indx],(Vx,Vy,V1x,V1y))

			omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes-1)/period[i] )
			angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - (jRes - 1) * meanLong[i]
			angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - (jRes - 1) * meanLong[i]

			TTVs[i] += period[i] / np.pi * ( Vx * np.sin(angleResInner) + Vy * np.cos(angleResInner) )
			TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter) + V1y * np.cos(angleResOuter) )

				
		
		unscaledTimes =  [ time + dt  for time,dt in zip(transitTimes,TTVs) ]
		
		return [ np.vstack((self.transitNumbers[i],times)).T for i,times in enumerate(unscaledTimes) ]
	
	def parameterAmplitudeTables(self,params):
		massesAndEccs = params[:3*self.nPlanets]
		
		mass,ex,ey = np.transpose(massesAndEccs.reshape(-1,3))
		periodsAndMeanLongs = np.hstack(( np.array( (1.0,2.0*ey[0] ) ),params[3*self.nPlanets:]))
		period,meanLong = np.transpose(periodsAndMeanLongs.reshape(-1,2))

		for pair,data in self.resonanceData.iteritems():
			i,j = pair
			print "planet pair %d / %d" % pair
			parameters = np.array(( mass[i],mass[j],ex[i],ey[i],ex[j],ey[j] ))
			periodRatio = period[j] / period[i]
			
			ttv1FSdata = self.complexTTVAmplitudes1FS(parameters,periodRatio,data)
			ttv0Fdata = self.complexTTVAmplitudes0F(parameters,periodRatio,data)
			ttv2Sdata = self.complexTTVAmplitudes0F(parameters,periodRatio,data)
			
			# Sum TTV contributions of 1F/1S terms
			print "(1 F/S): j\t Vx \t Vy \t V1x \t V1y"
			for entry in ttv1FSdata:
				jRes,Vx,Vy,V1x,V1y = entry
				print "(1 F/S): %d\t %.3g \t %.3g \t %.3g \t %.3g"%(jRes,Vx,Vy,V1x,V1y)
			print "---------------------------------------------------"
			print "(0F): j\t Vx \tV1x"
			for entry in ttv0Fdata:
				jRes,Vx,V1x = entry
				print "(0 F): %d\t %.3g \t %.3g"%(jRes,Vx,V1x)
			# Sum TTV contributions of 2S term
			print "(2S): j\t Vx \t Vy \t V1x \t V1y"
			jRes,Vx,Vy,V1x,V1y = ttv2Sdata
			print "(1 F/S): %d\t %.3g \t %.3g \t %.3g \t %.3g"%(jRes,Vx,Vy,V1x,V1y)
			


	def parameterTransitTimes(self,params,Only_1S=False):
		massesAndEccs = params[:3*self.nPlanets]
		ey0 = massesAndEccs[2]
		periodsAndMeanLongs = np.hstack(( np.array( (1.0, 0.0) ),params[3*self.nPlanets:]))
		#
		
		if Only_1S:
			unscaledTimes = [x[:,1] for x in self.TransitTimes_1SOnly(massesAndEccs,periodsAndMeanLongs)]
		else:
			unscaledTimes = [x[:,1] for x in self.TransitTimes(massesAndEccs,periodsAndMeanLongs)]
		
		#
		# time rescaling function
		def timeTransform(x,tau,t0):
			return tau * x + t0
		# Solve for the transform of analytic times that gives the best fit to observed transits
		tau,t0 = curve_fit(timeTransform, np.hstack(unscaledTimes), self.flatTimes,sigma = self.flatUncertainties)[0]		
		transform = np.vectorize(lambda x: timeTransform(x,tau,t0))

		return [ np.vstack((self.transitNumbers[i],transform(times))).T for i,times in enumerate(unscaledTimes) ]
	
	def parameterTTVPlot(self,params,**kwargs):
		fmt = kwargs.get('fmt','k.')
		transitNumberAndTime = self.parameterTransitTimes(params)
		for i,numAndTime in enumerate(transitNumberAndTime):
			pl.subplot(self.nPlanets*100 + 10 + i +1)
			ttv = linefit_resids(numAndTime[:,0],numAndTime[:,1])
			pl.plot(numAndTime[:,1],ttv,fmt)
	
	def parameterTTVResidualsPlot(self,params,normalized=False,**kwargs):
		fmt = kwargs.get('fmt','k.')
		transitNumberAndTime = self.parameterTransitTimes(params)
		for i,numAndTime in enumerate(transitNumberAndTime):
			pl.subplot(self.nPlanets*100 + 10 + i +1)
			resids = (self.transitTimes[i] - numAndTime[:,1])
			if normalized:
				resids /= self.transitUncertainties[i]
			ebs = self.transitUncertainties[i] 
			pl.errorbar(numAndTime[:,1],resids,yerr=ebs,fmt=fmt)
		
	def parameterTTV1SResidualsPlot(self,paramsList,**kwargs):
		fmt = kwargs.get('fmt','k.')
		for params in paramsList:
			transitNumberAndTime = self.parameterTransitTimes(params)
			transit1Sonly = self.parameterTransitTimes(params,Only_1S=True)
			for i,numAndTime in enumerate(transitNumberAndTime):
				pl.subplot(self.nPlanets*100 + 10 + i +1)
				resids = (numAndTime[:,1]-transit1Sonly[i][:,1])
				pl.plot(numAndTime[:,1],resids,fmt)

		for i in range(self.nPlanets):
			pl.subplot(self.nPlanets*100 + 10 + i +1)
			obstrNums = self.transitNumbers[i]
			obs_resids = self.transitTimes[i] - transit1Sonly[i][obstrNums,1]
			ebs = self.transitUncertainties[i]
			#print  self.transitTimes[i] - transit1Sonly[i][obstrNums,1]
			#print map(len,(obs_resids,ebs,transit1Sonly[i][:,1]))
			pl.errorbar(self.transitTimes[i],obs_resids,yerr=ebs,fmt='rs')

	def parameterFitness(self,params,Only_1S=False):

		transitNumberAndTime = self.parameterTransitTimes(params,Only_1S)
		chi2 = 0.0	
		for i in range(self.nPlanets):
			sigma = self.transitUncertainties[i]
			diff = transitNumberAndTime[i][:,1] - self.transitTimes[i]
			chi2 += np.sum( np.power(diff,2)/ np.power(sigma,2) )
		return -0.5 * chi2
				
	def bestFitParameters(self,params0):
		target_data = np.array([])
		errors = np.array([])
		for i in range(self.nPlanets):
			target_data = np.append(target_data,self.transitTimes[i])
			errors = np.append(errors,self.transitUncertainties[i])
		
		def objectivefn(x):
			transitNumberAndTime =  self.parameterTransitTimes(x)
			answer = np.array([],dtype=float)
			for t in transitNumberAndTime:
				answer = np.append( answer,np.array(t[:,1]) )
			#
			return (answer - target_data)/errors
		
		return leastsq(objectivefn, params0,full_output=1)

# ---------------------------	#	#	#	#	#	--------------------------- #

# if __name__=="__main__":
# 	with open('planets.txt') as fi:
# 		pNames = [line.strip() for line in fi.readlines()]
# 	analyticFit = MultiplanetAnalyticTTVSystem([np.loadtxt(pname) for pname in pNames])
# 	analyticFit.parameterTTV1SResidualsPlot(bestparams,fmt='k-')
if __name__=="__main__":
	import glob
	sys.path.append("/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface")
	import PyTTVFast as ttv

	# Read in parameters from 'inpars.txt' to generate transit times via n-body
	pars = np.loadtxt('inpars.txt')
	nbody_compute = ttv.TTVCompute()
	trTimes,success = nbody_compute.TransitTimes(200.,pars)
	
	inptData = []
	for times in trTimes:
		NandT=np.vstack(( np.arange(len(times)) , times , 1.e-5*np.ones(len(times)) )).T
		inptData.append(NandT)
	
	noiseLvl = 2.e-4
	pl.figure(1)
	for i,times in enumerate(trTimes):
		nTransits = len(times)
		noise = np.random.normal(0.,noiseLvl,nTransits)
		noisyData=np.vstack(( np.arange(nTransits) , times + noise , noiseLvl*np.ones(len(times)) )).T
		np.savetxt("planet%d.txt"%i,noisyData)
		pl.errorbar(noisyData[:,1],linefit_resids(noisyData[:,0],noisyData[:,1],noisyData[:,2]),yerr=noisyData[:,2],fmt='s')
		
	# Create an analytic fit object based on the N-body transit times
	analyticFit = MultiplanetAnalyticTTVSystem(inptData)

	# Convert parameters to form used by analytic fit
	mass = pars[:,0]
	ex  = pars[:,2] * np.cos(pars[:,5]*np.pi/180.)
	ey  = pars[:,2] * np.sin(pars[:,5]*np.pi/180.)	
	ex,ey = ey,-ex

	pers = pars[:,1]
	meanAnom = pars[:,-1]
	meanLong = np.pi*(meanAnom)/ 180. + np.arctan2(ey,ex) 
	massAndEccs = np.vstack((mass,ex,ey)).T
	persAndLs = np.vstack((  pers, np.mod(meanLong,2.*np.pi)   )).T

	# Generate analytic transit times from full parameter list

	transits = analyticFit.TransitTimes(massAndEccs,persAndLs)
			
	# Plot N-body and analytic transit times
	
	for i,timedata in enumerate(zip(transits,inptData)):
		times,obstimes = timedata
		pl.figure(2)
		pl.subplot(analyticFit.nPlanets*100+10 + i + 1)
		pl.plot(times[:,1])
		pl.plot(obstimes[:,1])
	
		pl.figure(3)	
		pl.subplot(analyticFit.nPlanets*100+10 + i + 1)
		ttvs = linefit_resids(times[:,0],times[:,1])
		pl.plot(times[:,1],ttvs,'k.')		
		obs_ttvs = linefit_resids(obstimes[:,0],obstimes[:,1])
		pl.errorbar(obstimes[:,1],obs_ttvs,yerr=obstimes[:,2],fmt='rs')

	# Compute and plot analytic transit times from abridged parameters (i.e., use time-rescaling method
	t0s = [x[0] for x in trTimes]
	Lvals =  -2 * np.pi/pers * ( t0s - t0s[0])
	transformedPersAndLs = np.vstack(( pers/pers[0], Lvals  )).T
	new_params=np.hstack((massAndEccs.reshape(-1),transformedPersAndLs.reshape(-1)[2:]))
	#
	new_transits = 	analyticFit.parameterTransitTimes(new_params)
	analyticFit.parameterTTVPlot(new_params,fmt='kx')
	
	pl.show()
	

	analyticFit.parameterAmplitudeTables(new_params)
	np.savetxt("starting_paramters.txt",new_params)

