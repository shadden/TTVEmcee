import numpy as np
from itertools import combinations
import LaplaceCoefficients as LC
from scipy.optimize import curve_fit,leastsq
import sys
sys.path.append("/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface")
import PyTTVFast as ttv

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
	
class MultiplanetSimpleAnalyticTTVSystem(object):
	def __init__(self,inputData,deltaLimit=0.1):

		# Store observed transit information
		self.nPlanets = len(inputData)
		self.observedData= inputData
		self.transitNumbers = [ (data[:,0]).astype(int) for data in self.observedData] 
		self.transitCounts = [len(dat) for dat in self.observedData]
		self.transitTimes = [ data[:,1] for data in self.observedData ]
		self.transitUncertainties = [ data[:,2] for data in self.observedData ] 
		
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
			j = get_res(p2/p1)
			pairDelta = delta(p2/p1,j,j-1.)
			
			if abs(pairDelta) < deltaLimit:
				# Get the Laplace coefficients for this pair
				laplaceCoeffF1ext = LC.get_f1Ext_array(p2/p1)[j-1]
				laplaceCoeffF1int = LC.get_f1Int_array(p2/p1)[j-1]
				laplaceCoeffF = LC.get_f_array(p2/p1)[j-1]
				resDataDict = {'j':j,'f':laplaceCoeffF,'f1e':laplaceCoeffF1ext,'f1i':laplaceCoeffF1int}
				self.resonanceData.update({pair:resDataDict})
		
	def complexTTVAmplitudes(self,massesAndEccs,periodRatio,resData):
		"""
		Get the first-order mean-motion resonance complex TTV amplitudes for a pair of planets.
		"""

		# Input variable parameters
		mu,mu1 = massesAndEccs[:2]
		ex,ey,ex1,ey1 = massesAndEccs[2:]
		
		# Input resonace information for pair
		j,f,f1e,f1i = map(lambda x: resData[x],('j','f','f1e','f1i'))
		delta = (j-1.) * periodRatio / j -1.
		alpha = np.power( periodRatio, -2./3.)

		# Complex eccentricities
		Zxe,Zxi = f * ex + f1e * ex1, f * ex + f1i * ex1
		Zye,Zyi = f * ey + f1e * ey1, f * ey + f1i * ey1

		Vx= mu1/np.sqrt(alpha) *  ( -f/(j * delta)  - (j-1.)/(j * alpha**1.5) * 1.5 * Zxe/delta**2 )
		Vy= mu1/np.sqrt(alpha) *  ( -(j-1.)/(j * alpha**1.5) * 1.5 * -Zye/delta**2 )
		V1x = mu*(-f1i/(j*delta) + 1.5* Zxi/(j*delta**2) )
		V1y = mu*(1.5 * -Zyi/(j*delta**2) )
		
		return Vx,Vy,V1x,V1y
	
	def TransitTimes(self,massesAndEccs,periodsAndMeanLongs):
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
			
			parameters = np.array(( mass[i],mass[j],ex[i],ey[i],ex[j],ey[j] ))
			periodRatio = period[j] / period[i]
			Vx,Vy,V1x,V1y = self.complexTTVAmplitudes(parameters,periodRatio,data)
			
			jRes = data['j']
			omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes-1)/period[i] )
			angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - (jRes - 1) * meanLong[i]
			angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - (jRes - 1) * meanLong[i]
			
			TTVs[i] += period[i] / np.pi * ( Vx * np.sin(angleResInner) + Vy * np.cos(angleResInner) )
			TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter) + V1y * np.cos(angleResOuter) )
		
		unscaledTimes =  [ time + dt  for time,dt in zip(transitTimes,TTVs) ]
		
		return [ np.vstack((self.transitNumbers[i],times)).T for i,times in enumerate(unscaledTimes) ]
		
	
	def parameterTransitTimes(self,params):
		massesAndEccs = params[:3*self.nPlanets]
		ey0 = massesAndEccs[2]
		periodsAndMeanLongs = np.hstack(( np.array( (1.0,2.0*ey0 ) ),params[3*self.nPlanets:]))
		unscaledTimes = [x[:,1] for x in self.TransitTimes(massesAndEccs,periodsAndMeanLongs)]
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
			subplot(self.nPlanets*100 + 10 + i +1)
			ttv = linefit_resids(numAndTime[:,0],numAndTime[:,1])
			plot(numAndTime[:,1],ttv,fmt)
	
	def parameterTTVResidualsPlot(self,params,normalized=False,**kwargs):
		fmt = kwargs.get('fmt','k.')
		transitNumberAndTime = self.parameterTransitTimes(params)
		for i,numAndTime in enumerate(transitNumberAndTime):
			subplot(self.nPlanets*100 + 10 + i +1)
			resids = (self.transitTimes[i] - numAndTime[:,1])
			if normalized:
				resids /= self.transitUncertainties[i]

			plot(numAndTime[:,1],resids,fmt)
		
	def parameterFitness(self,params):

		transitNumberAndTime = self.parameterTransitTimes(params)
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

	
	#	#	#	#	#
if __name__=="__main__":
	import glob

	pars = np.loadtxt('inpars.txt')
	nbody_compute = ttv.TTVCompute()
	trTimes,success = nbody_compute.TransitTimes(100.,pars)
	
	inptData = []
	for times in trTimes:
		NandT=np.vstack(( np.arange(len(times)) , times , 1.e-5*np.ones(len(times)) )).T
		inptData.append(NandT)
		
	analyticFit = MultiplanetSimpleAnalyticTTVSystem(inptData)


	mass = pars[:,0]
	ex  = pars[:,2] * np.cos(pars[:,5]*np.pi/180.)
	ey  = pars[:,2] * np.sin(pars[:,5]*np.pi/180.)	
	ex,ey = ey,-ex

	pers = pars[:,1]
	meanAnom = pars[:,-1]
	meanLong = np.pi*(meanAnom)/ 180. + np.arctan2(ey,ex) 
	massAndEccs = np.vstack((mass,ex,ey)).T
	persAndLs = np.vstack((  pers, np.mod(meanLong,2.*np.pi)   )).T
	transits = analyticFit.TransitTimes(massAndEccs,persAndLs)
		

	for i,timedata in enumerate(zip(transits,inptData)):
		times,obstimes = timedata
		figure(1)
		subplot(210 + i + 1)
		plot(times[:,1])
		plot(obstimes[:,1])
	
		figure(2)	
		subplot(210 + i + 1)
		ttvs = linefit_resids(times[:,0],times[:,1])
		plot(times[:,1],ttvs,'k.')		
		obs_ttvs = linefit_resids(obstimes[:,0],obstimes[:,1])
		errorbar(obstimes[:,1],obs_ttvs,yerr=obstimes[:,2],fmt='rs')

	t0s = [x[0] for x in trTimes]
	Lvals =  -2 * np.pi/pers * ( t0s - t0s[0])
	transformedPersAndLs = np.vstack(( pers/pers[0], Lvals  )).T
	new_params=hstack((massAndEccs.reshape(-1),transformedPersAndLs.reshape(-1)[2:]))
	analyticFit.parameterTTVPlot(new_params,fmt='kx')
	
	show()
	
	new_transits = 	analyticFit.parameterTransitTimes(new_params)
	
	print "expected first transits: "
	print -pers /(2. * np.pi) * meanLong
	print "observed first transits: "
	print trTimes[0][0],trTimes[1][0]
	print "analytic first transits (1): "
	print transits[0][0,1],transits[1][0,1]
	print "analytic first transits (2): "
	print new_transits[0][0,1],new_transits[1][0,1]

