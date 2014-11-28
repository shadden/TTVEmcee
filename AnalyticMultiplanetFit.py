import sys
sys.path.insert(0,'/Users/samuelhadden/13_HighOrderTTV/TTVEmcee')
import numpy as np
from itertools import combinations
import matplotlib.pyplot as pl
import LaplaceCoefficients as LC
from scipy.optimize import curve_fit,leastsq



def linefit(x,y,sigma=None):
	""" 
	Fit a line of the form y = a * x + b.   Uncertainties in y can be specifies as 'sigma'.
	Returns the results of 'curve_fit'.  The first return value is the tuplet (a,b).
	"""
	assert len(x) == len(y), "Cannot fit line with different length dependent and independent variable data!"
	linefn = lambda x,slope,intercept: x*slope + intercept
	return curve_fit(linefn,x,y,sigma=sigma)

def linefit_resids(x,y,sigma=None):
	"""
	Fit a line of the form y = a * x + b and return the residuals y- a * x - b.
	"""
	s,m = linefit(x,y,sigma)[0]
	return y - s*x -m

def delta(pratio,j,k):
	return pratio * k/j - 1.

def get_res(pratio):
	return np.argmin([np.abs(delta(pratio,j,j-1)) for j in range(2,6)]) + 2

def deg2rad(x):
	return np.mod( x * np.pi / 180. ,  2*np.pi)

def ttvfast2mycoords(coords):
	""" 
	Convert from ttvfast coordinates (positive z-axis pointed towards observer) to
	coordiantes where the observer is located along the positive x-axis
	
	Parameters
	----------
	coords : 2-dimensional array
		A list of mass and orbital data for each planet.  A single planet's entry is in 
		the form:
				[ Mass, Per, Ecc, Inc, Node, ArgPeri, M ]
				
	Returns
	-------
	new_coords: 2-dimensional array
		Similar to coords but transformed to the new coordinate system and each planet 
		entry is in the form:
			[ Mass,Per, Ecc, Inc, LongPeri, Node, MeanLong ]
	"""
	
	newCoords = np.zeros(coords.shape)
	
	for i,coord in enumerate(coords):
		Mass,Per,Ecc,Inc,Node,ArgPeri,M = coord
		Inc = deg2rad(Inc-90.) 
		LongPeri = deg2rad(ArgPeri + Node - 90.)
		MeanLong = np.mod( deg2rad(M) + LongPeri, 2.*np.pi)
		Node = deg2rad(Node)
		newCoords[i] = np.array( (Mass,Per, Ecc, Inc, LongPeri, Node, MeanLong) )

	return newCoords

def orbels2parameters(coords):
	massAndEcc = np.zeros( (coords.shape[0], 3))
	perAndL = np.zeros( (coords.shape[0], 2))
	for i,coord in enumerate(coords):
		Mass,Per, Ecc, Inc, LongPeri, Node, MeanLong = coord
		ex = Ecc * np.cos(LongPeri)
		ey = Ecc * np.sin(LongPeri)
		massAndEcc[i]=np.array((Mass,ex,ey))
		perAndL[i] = np.array((Per,MeanLong))

	return massAndEcc,perAndL 

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
				'df':laplaceCoeffdF,'df1e':laplaceCoeffdF1ext,'df1i':laplaceCoeffdF1int,\
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
		j = np.arange(1,LC.jMax)
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
		
		#print fe[1],f1e[1],ey,ey1,Zye[1]
		 
		Vx= mu1/np.sqrt(alpha) * (-fe/(j * delta) -(j-1.)/j * 1.5/(j*alpha**1.5 * delta**2 ) * Zxe)
		Vy= mu1/np.sqrt(alpha) *  ( -(j-1.)/j * 1.5/(j * alpha**1.5 * delta**2 )) * -Zye
		#-- extra lambda terms --#
		Vx+= -mu1*np.sqrt(alpha)/(j*delta) *  ((dfe - 0.25*f/np.sqrt(alpha)) *  ex + df1e *  ex1 )
		Vy+= -mu1*np.sqrt(alpha)/(j*delta) *  ((dfe - 0.25*f/np.sqrt(alpha)) * -ey + df1e * -ey1 )
		
		
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
		j = np.arange(1,LC.jMax)
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
		dlx = -mu1 * 1.5/(j*delta**2) * (j-2.)/(j*alpha**2) * bigZ2x #
		dly = -mu1 * 1.5/(j*delta**2) *(j-2.)/(j*alpha**2) * -bigZ2y
		dl1x = mu * 1.5/(j*delta**2) * bigZ2x
		dl1y = mu * 1.5/(j*delta**2) * -bigZ2y
		#--------------------------------------------		
		return np.array((j,dzx + dlx, dzy + dly, dz1x + dl1x, dz1y + dl1y))
	
	def ContinuousTransitTimes(self,massesAndEccs,periodsAndMeanLongs,Npts=200,exclude=[],epoch = 0.0):
		excludeF =  ( 'F' in exclude )
		exclude2S =  ( '2S' in exclude )			
		mass,ex,ey = np.transpose(massesAndEccs.reshape(-1,3))
		period,meanLong = np.transpose(periodsAndMeanLongs.reshape(-1,2))
		tInit = 0.5 * period *(-meanLong +2*ey) / np.pi 
		continuousIndicies = [ np.linspace(0.,np.max(Nums),Npts) for Nums in self.transitNumbers ]
			
		transitTimes = [ (period[i] * continuousIndicies[i] + tInit[i]) for i in range(self.nPlanets) ]
		TTVs  = [ np.zeros(Npts) for i in range(self.nPlanets) ]

		for pair,data in self.resonanceData.iteritems():
			i,j = pair
			
			parameters = np.array(( mass[i],mass[j],ex[i],ey[i],ex[j],ey[j] ))
			periodRatio = period[j] / period[i]
			
			ttv1FSdata = self.complexTTVAmplitudes1FS(parameters,periodRatio,data)
			ttv0Fdata = self.complexTTVAmplitudes0F(parameters,periodRatio,data)
			
			# Sum TTV contributions of 1F/1S terms
			for entry in ttv1FSdata:
				
				jRes,Vx,Vy,V1x,V1y = entry

				if excludeF and jRes != data['j'] or np.abs(data['Delta']) > 0.08:
					continue

				omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes-1)/period[i] )
				angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - (jRes - 1) * meanLong[i]
				angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - (jRes - 1) * meanLong[i]

				TTVs[i] += period[i] / np.pi * ( Vx * np.sin(angleResInner) + Vy * np.cos(angleResInner) )
				TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter) + V1y * np.cos(angleResOuter) )
			
			# Sum TTV contributions of 0F terms
			for entry in ttv0Fdata:
				
				if excludeF:
					continue
				
				jRes,Vx,V1x = entry

				omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes)/period[i] )
				angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - jRes * meanLong[i]
				angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - jRes * meanLong[i]

				TTVs[i] += period[i] / np.pi * ( Vx * np.sin(angleResInner) )
				TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter))

			# Add contribution of 2S term if delta is small
			if np.abs(data['Delta']) < self.deltaLimit and not exclude2S:
				jRes,Vx,Vy,V1x,V1y = self.complexTTVAmplitudes2S(parameters,periodRatio,data)
				omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes-2)/period[i] )
				angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - (jRes - 2) * meanLong[i]
				angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - (jRes - 2) * meanLong[i]
				TTVs[i] += period[i] / np.pi * ( Vx  * np.sin(angleResInner) + Vy  * np.cos(angleResInner) )
				TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter) + V1y * np.cos(angleResOuter) )

				
		
		unscaledTimes =  [ epoch +time + dt  for time,dt in zip(transitTimes,TTVs) ]
		
		return [ np.vstack((continuousIndicies[i],times)).T for i,times in enumerate(unscaledTimes) ]

	def TransitTimes(self,massesAndEccs,periodsAndMeanLongs,exclude=[],epoch = 0.0):
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
		exclude : list
			List of terms to exclude from TTV calculation.  Options and 'F' for fast terms and'2S' for
			the second-harmonic slow term.
		"""
		excludeF =  ( 'F' in exclude )
		exclude2S =  ( '2S' in exclude )
			
		mass,ex,ey = np.transpose(massesAndEccs.reshape(-1,3))
		period,meanLong = np.transpose(periodsAndMeanLongs.reshape(-1,2))

		tInit = 0.5 * period *(-meanLong + 2*ey) / np.pi 
		
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

				if excludeF and jRes != data['j']:
					continue

				omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes-1)/period[i] )
				angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - (jRes - 1) * meanLong[i]
				angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - (jRes - 1) * meanLong[i]

				TTVs[i] += period[i] / np.pi * ( Vx * np.sin(angleResInner) + Vy * np.cos(angleResInner) )
				TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter) + V1y * np.cos(angleResOuter) )
			
			# Sum TTV contributions of 0F terms
			for entry in ttv0Fdata:
				
				if excludeF:
					continue
				
				jRes,Vx,V1x = entry

				omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes)/period[i] )
				angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - jRes * meanLong[i]
				angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - jRes * meanLong[i]

				TTVs[i] += period[i] / np.pi * ( Vx * np.sin(angleResInner) )
				TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter))

			# Add contribution of 2S term if delta is small
			if np.abs(data['Delta']) < self.deltaLimit and not exclude2S:
				jRes,Vx,Vy,V1x,V1y = self.complexTTVAmplitudes2S(parameters,periodRatio,data)
				omegaRes = 2.0 * np.pi * ( jRes/period[j] - (jRes-2)/period[i] )
				angleResInner = omegaRes * transitTimes[i] + jRes * meanLong[j] - (jRes - 2) * meanLong[i]
				angleResOuter = omegaRes * transitTimes[j] + jRes * meanLong[j] - (jRes - 2) * meanLong[i]
				TTVs[i] += period[i] / np.pi * ( Vx  * np.sin(angleResInner) + Vy  * np.cos(angleResInner) )
				TTVs[j] += period[j] / np.pi * ( V1x * np.sin(angleResOuter) + V1y * np.cos(angleResOuter) )

				
		
		unscaledTimes =  [ epoch + time + dt  for time,dt in zip(transitTimes,TTVs) ]
		
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
			ttv2Sdata = self.complexTTVAmplitudes2S(parameters,periodRatio,data)
			
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
			print "---------------------------------------------------"
			print "(2S): j\t Vx \t Vy \t V1x \t V1y"
			jRes,Vx,Vy,V1x,V1y = ttv2Sdata
			print "(2S): %d\t %.3g \t %.3g \t %.3g \t %.3g"%(jRes,Vx,Vy,V1x,V1y)
			


	def parameterTransitTimes(self,params,Only_1S=False,exclude=[],return_transform=False):
		massesAndEccs = params[:3*self.nPlanets]
		ey0 = massesAndEccs[2]
		periodsAndMeanLongs = np.hstack(( np.array( (1.0, 0.0) ),params[3*self.nPlanets:]))
		#
		
		if Only_1S:
			unscaledTimes = [x[:,1] for x in self.TransitTimes_1SOnly(massesAndEccs,periodsAndMeanLongs)]
		else:
			unscaledTimes = [x[:,1] for x in self.TransitTimes(massesAndEccs,periodsAndMeanLongs,exclude)]
		
		#
		# time rescaling function
		def timeTransform(x,tau,t0):
			return tau * x + t0
		# Solve for the transform of analytic times that gives the best fit to observed transits
		tau,t0 = curve_fit(timeTransform, np.hstack(unscaledTimes), self.flatTimes,sigma = self.flatUncertainties)[0]		
		transform = np.vectorize(lambda x: timeTransform(x,tau,t0))
		
		if return_transform:
			return [ np.vstack((self.transitNumbers[i],transform(times))).T for i,times in enumerate(unscaledTimes) ],transform
			
		return [ np.vstack((self.transitNumbers[i],transform(times))).T for i,times in enumerate(unscaledTimes) ]
	
##########################################################################################
#	Begin TTV plot	
##########################################################################################
	def parameterTTVPlot(self,params,**kwargs):
		"""
		Plot the TTVs from a list of different input parameters as well as the observed TTVs and their errorbars.
		Parameters
		----------
		params : array
		 List of parameters to analytic model for plotting.  Either an N x (5*nPlanets-2)  2-dimensional array
		or a single set of (3*nPlanets-2) paramters.
		"""
		exclude = kwargs.get('exclude',[])
		sparams = params.reshape(-1,self.nPlanets*5-2)
		axList = []
		#----------------------- Observed TTVs -----------------------#
		for i,data in enumerate(zip(self.transitNumbers,self.transitTimes,self.transitUncertainties)):
			if i==0:
				axList.append( pl.subplot(self.nPlanets*100 + 10 + i +1) )
			else:
				axList.append( pl.subplot(self.nPlanets*100 + 10 + i +1,sharex=axList[0]) )
			if i != self.nPlanets-1:
					pl.setp( axList[i].get_xticklabels(), visible=False )

			trNums,trTimes,trErrs = data 
			#pl.subplot(self.nPlanets*100 + 10 + i +1)
			
			ttv = linefit_resids(trNums,trTimes,trErrs)
			axList[i].errorbar(trTimes,ttv,yerr=trErrs,fmt='kx')
		
		pl.subplots_adjust(hspace=0.0)
		
		#----------------------- Analytic TTVs -----------------------#
		for param in sparams:
			fmt = kwargs.get('fmt','.')
			transitNumberAndTime = self.parameterTransitTimes(param,exclude=exclude)
			for i,numAndTime in enumerate(transitNumberAndTime):
				#pl.subplot(self.nPlanets*100 + 10 + i +1)
				ttv = linefit_resids(numAndTime[:,0],numAndTime[:,1])
				axList[i].plot(numAndTime[:,1],ttv,fmt)
		
		
		for ax in axList:
			yticks = ax.get_yticks()
			ax.set_yticks(yticks[1:-1])
##########################################################################################
#	End TTV plot	
##########################################################################################


##########################################################################################
#	Begin Continuous TTV Plot	
##########################################################################################
	def parameterContinuousPlot(self,params,**kwargs):
		"""
		Plot the 1S sinusoid from a list of different input parameters as well as the observed TTVs and their errorbars.
		Parameters
		----------
		params : array
		 List of parameters to analytic model for plotting.  Either an N x (5*nPlanets-2)  2-dimensional array
		or a single set of (3*nPlanets-2) paramters.
		"""
		axList = []
		exclude = kwargs.get('exclude',[])	
		Npts = kwargs.get('Npts',200)
		#----------------------- Analytic TTVs -----------------------#

		mass_ecc = params[:self.nPlanets * 3]
		pAndL = self.bestFitUnscaledPeriodAndLongitude(mass_ecc)
		continuousTimes = self.ContinuousTransitTimes(mass_ecc,pAndL ,epoch=np.min(self.flatTimes),exclude=exclude,Npts=Npts )
		aTimes = self.TransitTimes(mass_ecc,pAndL ,epoch=np.min(self.flatTimes) )

		#----------------------- Observed TTVs -----------------------#
		for i,data in enumerate(zip(self.transitNumbers,self.transitTimes,self.transitUncertainties)):
			if i==0:
				axList.append( pl.subplot(self.nPlanets*100 + 10 + i +1) )
			else:
				axList.append( pl.subplot(self.nPlanets*100 + 10 + i +1,sharex=axList[0]) )
			if i != self.nPlanets-1:
					pl.setp( axList[i].get_xticklabels(), visible=False )

			trNums,trTimes,trErrs = data 
			#pl.subplot(self.nPlanets*100 + 10 + i +1)
			
			ttv = linefit_resids(trNums,trTimes,trErrs)
			ttv = trTimes - trNums * self.periodEstimates[i] - self.tInitEstimates[i]
			axList[i].errorbar(trTimes,ttv,yerr=trErrs,fmt='kx',mew=1.3)
			continuousTTV=continuousTimes[i][:,1] - continuousTimes[i][:,0] * self.periodEstimates[i] - self.tInitEstimates[i]
			axList[i].plot(continuousTimes[i][:,1],continuousTTV,'k-')
			aTTV = aTimes[i][:,1] - aTimes[i][:,0] * self.periodEstimates[i] - self.tInitEstimates[i]
			axList[i].plot(aTimes[i][:,1],aTTV,linestyle='',marker='o',mfc='w',mec='r',mew=2,zorder=1)
			
			maxTTV = np.max( np.abs(ttv) )
			axList[i].set_ylim( -1.3*maxTTV, 1.3*maxTTV )
			
			locs,labls = yticks()
			locs = map(lambda x: int(round(24*60*x))/(24.*60.),locs[1:-1])
			yticks( locs , 24.*60.*np.array(locs) ) 
		pl.subplots_adjust(hspace=0.0)
			
		return(axList)
##########################################################################################
#	End Continous TTV Plot	
##########################################################################################
	
	
	
	def parameterTTVResidualsPlot(self,params,normalized=False,**kwargs):
		fmt = kwargs.get('fmt','k.')
		sparams = params.reshape(-1,self.nPlanets*5-2)
		exclude = kwargs.get('exclude',[])

		transitNumberAndTime = self.parameterTransitTimes(params,exclude=exclude)
		for i,numAndTime in enumerate(transitNumberAndTime):
			pl.subplot(self.nPlanets*100 + 10 + i +1)
			resids = (self.transitTimes[i] - numAndTime[:,1])
			if normalized:
				resids /= self.transitUncertainties[i]
			ebs = self.transitUncertainties[i] 
			pl.errorbar(numAndTime[:,1],resids,yerr=ebs,fmt=fmt)
		
	def parameterTTV1SResidualsPlot(self,paramsList,**kwargs):
		paramsList = paramsList.reshape(-1,self.nPlanets*5 -2)
		fmt = kwargs.get('fmt','.')
		exclude = kwargs.get('exclude',[])
		showObs = kwargs.get('showObs',True)
		label = kwargs.get('label',None)
		print label
		axList = []
		# Set up axes
		
		for i in range(self.nPlanets):
			if i==0:
				axList.append( pl.subplot(self.nPlanets*100 + 10 + i +1) )
			else:
				axList.append( pl.subplot(self.nPlanets*100 + 10 + i +1,sharex=axList[0]) )
			if i != self.nPlanets-1:
					pl.setp( axList[i].get_xticklabels(), visible=False )
			#axList[i].yaxis.set_major_locator( MaxNLocator(8) )
			#pl.subplot(self.nPlanets*100 + 10 + i +1)
		pl.subplots_adjust(hspace=0.0)
		
		for params in paramsList:
			
			transitNumberAndTime = self.parameterTransitTimes(params,exclude=exclude)
			transit1Sonly = self.parameterTransitTimes(params,Only_1S=True)
			
			for i,numAndTime in enumerate(transitNumberAndTime): 
			
				resids = (numAndTime[:,1]-transit1Sonly[i][:,1])
				axList[i].plot(numAndTime[:,1],resids,fmt,label=label)
			
				if  showObs:
					obs_resids = self.transitTimes[i] - transit1Sonly[i][:,1]
					ebs = self.transitUncertainties[i]
					axList[i].errorbar(self.transitTimes[i],obs_resids,yerr=ebs,fmt='kx',label='Observed')
		
		return axList



	def parameterFitness(self,params,Only_1S=False,exclude=[]):

		transitNumberAndTime = self.parameterTransitTimes(params,Only_1S,exclude)
		chi2 = 0.0	
		for i in range(self.nPlanets):
			sigma = self.transitUncertainties[i]
			diff = transitNumberAndTime[i][:,1] - self.transitTimes[i]
			chi2 += np.sum( np.power(diff,2)/ np.power(sigma,2) )
		return -0.5 * chi2
				
	def bestFitParameters(self,params0,exclude=[]):
		target_data = np.array([])
		errors = np.array([])
		for i in range(self.nPlanets):
			target_data = np.append(target_data,self.transitTimes[i])
			errors = np.append(errors,self.transitUncertainties[i])
		
		def objectivefn(x):
			transitNumberAndTime =  self.parameterTransitTimes(x,exclude=exclude)
			answer = np.array([],dtype=float)
			for t in transitNumberAndTime:
				answer = np.append( answer,np.array(t[:,1]) )
			#
			return (answer - target_data)/errors
		
		return leastsq(objectivefn, params0,full_output=1)

	def bestFitPeriodAndLongitude(self,massAndEcc,return_transform=False):
		""" 
		Find the best set of average periods and initial mean longitudes for a fixed set of 
		planet masses and eccentricities.
	
		Parameters
		----------
		massAndEcc : nPlanets x 3 array
			The mass and free eccentricity vectors of each planet under consideration
		"""
	
		target_data = np.array([])
		errors = np.array([])
		
		for i in range(self.nPlanets):
			target_data = np.append(target_data,self.transitTimes[i])
			errors = np.append(errors,self.transitUncertainties[i])
	
		def objectivefn(x):
			fullParameters = np.hstack((massAndEcc.reshape(-1) , x))
			transitNumberAndTime =  self.parameterTransitTimes(fullParameters)
			answer = np.array([],dtype=float)
			for t in transitNumberAndTime:
				answer = np.append( answer,np.array(t[:,1]) )
		
			return (answer - target_data)/errors
		
		guess = np.vstack(( self.periodEstimates[1:]/self.periodEstimates[0] , np.zeros(self.nPlanets - 1) )).T.reshape(-1)
		
		pAndLbest =  leastsq(objectivefn, guess ,full_output=1)[0]

		if return_transform:
			pars = np.hstack((massAndEcc.reshape(-1),pAndLbest))
			transform=self.parameterTransitTimes(pars,return_transform = True)[-1]
			return pars,transform
			
		return np.hstack((massAndEcc.reshape(-1),pAndLbest))
	
	def bestFitUnscaledPeriodAndLongitude(self,pars):
		""" 
		Find the best set of average periods and initial mean longitudes for a fixed set of 
		planet masses and eccentricities.
	
		Parameters
		----------
		massAndEcc : nPlanets x 3 array
			The mass and free eccentricity vectors of each planet under consideration
		"""
		parlen =len(pars.reshape(-1))
		massAndEcc = pars[:3*self.nPlanets]
		target_data = np.array([])
		errors = np.array([])
		
		for i in range(self.nPlanets):
			target_data = np.append(target_data,self.transitTimes[i])
			errors = np.append(errors,self.transitUncertainties[i])
	
		epoch = np.min(self.flatTimes) 
		def objectivefn(x):
			transitNumberAndTime =  self.TransitTimes(massAndEcc,x, epoch=epoch )
			answer = np.array([],dtype=float)
			for t in transitNumberAndTime:
				answer = np.append( answer,np.array(t[:,1]) )
		
			return (answer - target_data)/errors
		
		pGuess = self.periodEstimates
		
		lambdaGuess = -2*np.pi*(self.tInitEstimates - epoch) / pGuess
		
		guess = np.vstack(( pGuess, lambdaGuess )).T.reshape(-1)
		bestSoln =  leastsq(objectivefn, guess ,full_output=1)[0]

		return bestSoln

###############################################################################################################################
	
	def forcedEccs(self,massesAndEccs,periodsAndMeanLongs):
		
		mass,ex,ey = np.transpose(massesAndEccs.reshape(-1,3))
		period,meanLong = np.transpose(periodsAndMeanLongs.reshape(-1,2))
		
		exF,eyF = np.zeros((2,self.nPlanets))
		for pair,data in self.resonanceData.iteritems():
			i,j = pair
			periodRatio = period[j] / period[i]
			alpha = np.power(periodRatio , -2./3.)
			jRes = data['j']
			
			fCoeff,f1Coeff,delta = data['f'][jRes],data['f1i'][jRes],data['Delta']
			
			resAngle = jRes * meanLong[j] - (jRes - 1.) * meanLong[i]
			eFin  = -mass[j] / np.sqrt(alpha) * 1./(jRes * delta) * fCoeff
			eFout = -mass[i] * 1. / (jRes * delta) * f1Coeff

			exF[i] += eFin  * np.cos(resAngle)
			eyF[i] += eFin  * np.sin(resAngle) 
			exF[j] += eFout * np.cos(resAngle)
			eyF[j] += eFout * np.sin(resAngle) 
			
		return np.vstack((exF,eyF)).T,np.vstack((ex-exF,ey-eyF)).T
			
	def TTVFastCoordTransform(self,ttvFastCoords):
		# if the input is given as a list of elements in form [[mass,period,ecc,...],...]
		
		if ttvFastCoords.shape[-1] == 7:
			# Transform to observer along x-axis:
			tCoords = ttvfast2mycoords(ttvFastCoords)
			# Get parameters in eccentricity component form:
			massAndEccentricities,periodsAndLongitudes = orbels2parameters(tCoords)
			periodsAndLongitudes[:,0] = self.periodEstimates
		else:
			coordCopy = ttvFastCoords.copy()
			massAndEccentricities = coordCopy[:3*self.nPlanets].reshape(-1,3)
			massAndEccentricities[:,1],massAndEccentricities[:,2] = massAndEccentricities[:,2],-massAndEccentricities[:,1]
			periodsAndLongitudes = self.bestFitPeriodAndLongitude(massAndEccentricities.reshape(-1,3))[-2*(self.nPlanets-1):]
			periodsAndLongitudes = np.vstack((np.array([1.0,0.0]), periodsAndLongitudes.reshape(-1,2) ))
		# convert total eccentricities to free eccentricities:
		
		eForced,eFree = self.forcedEccs(massAndEccentricities,periodsAndLongitudes)
		mass = massAndEccentricities[:,0].reshape(-1,1)
		# Replace instantaneous periods with approximate average period
		
		return np.hstack((mass, eFree)), periodsAndLongitudes
		
##############################################################################################################################################################################

if __name__=="__main__":
	with open('planets.txt') as fi:
		plfiles = [line.strip() for line in fi.readlines()]
	noisyData= [ np.loadtxt(planet) for planet in plfiles ]
	bestpars= loadtxt('bestpars.txt')
	
	analyticFit = MultiplanetAnalyticTTVSystem(noisyData)
	mAnde,pAndl = analyticFit.TTVFastCoordTransform(pars)
	
	pAndLbest = analyticFit.bestFitPeriodAndLongitude(mAnde)
	analyticFit.bestFitUnscaledPeriodAndLongitude(mAnde)
	axList=analyticFit.parameterContinuousPlot(mAnde,exclude=['F','2S'],Npts=300)

	show()
	
#  analyticFit.parameterTTV1SResidualsPlot(pAndLbest,label='Full Analytic Model',fmt='bs-')
#  axList = analyticFit.parameterTTV1SResidualsPlot(pAndLbest,exclude=['F'],showObs=False,label='No Fast Terms',fmt='rs-')
#  axList[-1].legend(loc=3)
# 
#  xlabel('Time')
#  ylabel('TTV')
#  sca(axList[0])
#  for ax in axList:
# 			ax.set_yticks( ax.get_yticks()[1:-1] )
#  title('Kepler-114 TTV Sinusoid Model Residuals',fontsize=20)
#  
#  show()
# ---------------------------	#	#	#	#	#	--------------------------- #

# if __name__=="__main__":
# 	with open('planets.txt') as fi:
# 		pNames = [line.strip() for line in fi.readlines()]
# 	analyticFit = MultiplanetAnalyticTTVSystem([np.loadtxt(pname) for pname in pNames])
# 	analyticFit.parameterTTV1SResidualsPlot(bestparams,fmt='k-')


#------------------------------# Garbage Code! #------------------------------#

# 	mass = pars[:,0]
# 	ex  = pars[:,2] * np.cos(pars[:,5]*np.pi/180.)
# 	ey  = pars[:,2] * np.sin(pars[:,5]*np.pi/180.)	
# 	ex,ey = ey,-ex
# 
# 	pers = pars[:,1]
# 	meanAnom = pars[:,-1]
# 	meanLong = np.pi*(meanAnom)/ 180. + np.arctan2(ey,ex) 
# 	massAndEccs = np.vstack((mass,ex,ey)).T
# 	persAndLs = np.vstack((  pers, np.mod(meanLong,2.*np.pi)   )).T

	# Compute and plot analytic transit times from abridged parameters ( i.e., use time-rescaling method )
# 	t0s = [x[0] for x in trTimes]
# 	Lvals =  -2 * np.pi/pers * ( t0s - t0s[0])
# 	transformedPersAndLs = np.vstack(( pers/pers[0], Lvals  )).T
# 	new_params=np.hstack((massAndEccs.reshape(-1),transformedPersAndLs.reshape(-1)[2:]))
# 	#
# 	new_transits = 	analyticFit.parameterTransitTimes(new_params)
# # 	analyticFit.parameterTTVPlot(new_params,fmt='kx')
	

#------------------------------# Garbage Code! #------------------------------#

			
