#  A collection of functions to compute priors on inclination
#   based on various observed transit features.
import numpy as np

# Solar radius in AU
r2au = 0.004649

def guassian_prior(inclinations, observed, sigma):
	return -0.5*np.sum((inclinations - observed)**2/sigma**2)
	
def ImpactParametersPriors(inclinations, periods, b, rstar, mstar, sigma_b,sigma_rstar,sigma_mstar)
	""" Priors based on formula (a/R_*) cos(i) = b """
	cosi = np.abs(np.cos(inclinations))
	a = ( mstar * (period / 365.25)**2 )**(1./3.)
	
	cosi0 = np.abs( b * rstar * r2au / a )
	
	#dcosi ~ (1/3) * da/a + dr/r + db/b 
 	sigma_cosi = np.sqrt( (sigma_rstar/rstar)**2 + (sigma_mstar/(3 *mstar) )**2 + (sigma_b/b)**2 )
 	
 	return -0.5 * np.sum ( (cosi - cosi0)**2 / sigma_cosi**2 )
