#  A collection of functions to compute priors on inclination
#   based on various observed transit features.
import numpy as np

# Solar radius in AU
r2au = 0.004649

def guassian_prior(inclinations, observed, sigma):
	return -0.5*np.sum((inclinations - observed)**2/sigma**2)
	
def ImpactParametersPriors(inclinations, periods, b, rstar, mstar, sigma_b,sigma_rstar,sigma_mstar):
	""" 
		Gaussian priors based on formula (a/R_*) cos(i) = b.  Assumes that fractional uncertainties
		are sufficiently small that we can linearize cos(i)'s dependence on the parameter errors
		and therefore add error contributions in quadrature.
	"""
	rstar *=  r2au
	sigma_rstar *= r2au
	
	cosi = np.abs(np.cos(inclinations))
	a = ( mstar * (periods/ 365.25)**2 )**(1./3.)
	
	cosi0 = np.abs( b * rstar / a )
	
	#sigma_cosi ~  Sum [ d (cosi) /dq * sigma_q ]
	dcos_dM = -(1./3.) * cosi0  / mstar
	dcos_dr = b / a
	dcos_db = rstar / a
	
 	sigma_cosi = np.sqrt( (dcos_dr * sigma_rstar)**2 + (dcos_dM * sigma_mstar )**2 + (dcos_db * sigma_b )**2 )

 	return -0.5 * np.sum ( (cosi - cosi0)**2 / sigma_cosi**2 )

def ImpactParametersToInclinations(periods, b, rstar, mstar):
	""" 
		Convert periods and impact parameters to inclinations for a given stellar radius and mass
	"""
	rstar *= r2au
	
	a = ( mstar * (periods/ 365.25)**2 )**(1./3.)
	
	cosi0 = np.abs( b * rstar / a )
 	
 	return np.arccos(cosi0)
