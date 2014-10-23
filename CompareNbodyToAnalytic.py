
import glob
import sys

sys.path.append("/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface")
sys.path.append("/Users/samuelhadden/13_HighOrderTTV/TTVEmcee/")

import PyTTVFast as nTTV
import AnalyticMultiplanetFit as aTTV
import numpy as np
import matplotlib.pyplot as pl

# Read in parameters from 'inpars.txt' and generate transit times via n-body
pars = np.loadtxt('inpars.txt')
nbody_compute = nTTV.TTVCompute()
trTimes,success = nbody_compute.TransitTimes(200.,pars)

inptData = []

for times in trTimes:
	NandT=np.vstack(( np.arange(len(times)) , times , 1.e-5*np.ones(len(times)) )).T
	inptData.append(NandT)

noiseLvl = 0.5e-4
#pl.figure(1)
noisyData = []
noiseTotal = 0.0
for i,times in enumerate(trTimes):
	nTransits = len(times)
	noise = np.random.normal(0.,noiseLvl,nTransits)
	noisyData.append(np.vstack(( np.arange(nTransits) , times + noise , noiseLvl*np.ones(len(times)) )).T)
	noiseTotal+= -0.5*np.sum( noise**2 / noiseLvl**2 )

#	np.savetxt("planet%d.txt"%i,noisyData)
#	pl.errorbar(noisyData[:,1],linefit_resids(noisyData[:,0],noisyData[:,1],noisyData[:,2]),yerr=noisyData[:,2],fmt='s')

# Create an analytic fit object based on the noisy N-body transit times
analyticFit = aTTV.MultiplanetAnalyticTTVSystem(noisyData)
nbodyFit = nTTV.TTVFitnessAdvanced(noisyData)

# Convert parameters to form used by analytic fit
mAnde,pAndl = analyticFit.TTVFastCoordTransform(pars)

# Generate analytic transit times for the 'true' masses and eccentricities
pAndLbest = analyticFit.bestFitPeriodAndLongitude(mAnde)
chi2true = analyticFit.parameterFitness(pAndLbest)
chi2true1sOnly = analyticFit.parameterFitness(pAndLbest,Only_1S=True)
transits = analyticFit.parameterTransitTimes(pAndLbest)


# Find the best-fit masses and eccentricities for the analytic model and determine the transit times
best_params = analyticFit.bestFitParameters(pAndLbest)[0]
chi2best = analyticFit.parameterFitness(best_params)
chi2best1sOnly = analyticFit.parameterFitness(best_params,Only_1S=True)
bestTransits = analyticFit.parameterTransitTimes(best_params)
	

# Find best-fit masses and eccentricities for the nbody model and determine the times
mass = pars[:,0]
period = pars[:,1]
eccentricity = pars[:,2]
argPeri = aTTV.deg2rad(pars[:,5])
meanAnom = aTTV.deg2rad(pars[:,6])
ex = eccentricity * np.cos(argPeri)
ey = eccentricity * np.sin(argPeri)
meanLong = meanAnom + argPeri 
# Time of inner planet's first transit:
T0 = noisyData[0][0,1]
tmp = np.vstack(( period[1:]/period[0] ,  meanLong[1:] - period[0]/period[1:] * meanLong[0] )).T
coplanarPars0 = np.append( np.vstack((mass,ex,ey)).T.reshape(-1) , tmp.reshape(-1) )

bestNbody = nbodyFit.CoplanarParametersTTVFit(coplanarPars0)[0]
chi2nbest = nbodyFit.CoplanarParametersFitness(bestNbody)
bestNtransits = nbodyFit.CoplanarParametersTransformedTransits(bestNbody,observed_only=True)[0]
bestNtransits = [np.vstack((np.arange(len(x)),x)).T for x in bestNtransits ]

###################################################################################
# Plot N-body and analytic transit times

for i,timedata in enumerate(zip(transits,bestTransits,bestNtransits,noisyData)):
	times,bestTimes,bestNtimes,obstimes = timedata

	# Plot the TTVs
	pl.figure(1)	
	pl.subplot(analyticFit.nPlanets*100+10 + i + 1)
	
	ttvs = aTTV.linefit_resids(times[:,0],times[:,1])
	bestAttvs = aTTV.linefit_resids(bestTimes[:,0],bestTimes[:,1])
	bestNttvs = aTTV.linefit_resids(bestNtimes[:,0],bestNtimes[:,1])

	obs_ttvs = aTTV.linefit_resids(obstimes[:,0],obstimes[:,1])
	
	
	pl.plot(times[:,1],ttvs,'g.')
	pl.plot(bestTimes[:,1],bestAttvs,'r.')
	pl.plot(bestNtimes[:,1],bestNttvs,'b.')

	pl.errorbar(obstimes[:,1],obs_ttvs,yerr=obstimes[:,2],fmt='ks')
	
	
	
	# Plot the TTV residuals	
	pl.figure(2)	
	pl.subplot(analyticFit.nPlanets*100+10 + i + 1)
	
	resids = obstimes[:,1] - times[:,1]
	bestResids = obstimes[:,1] - bestTimes[:,1]
	bestNresids = obstimes[:,1] - bestNtimes[:,1]
	
	pl.errorbar(obstimes[:,1],resids,yerr=obstimes[:,2],fmt='gs')
	pl.errorbar(obstimes[:,1],bestResids,yerr=obstimes[:,2],fmt='rs')
	pl.errorbar(bestNtimes[:,1],bestNresids,yerr=obstimes[:,2],fmt='bs')

pl.show()
nbFreeEcc = analyticFit.forcedEccs(np.array((bestNbody[0],bestNbody[2],-bestNbody[1], bestNbody[3], bestNbody[5], -bestNbody[4])),pAndl)[1].reshape(-1)

print "Free eccentricity comparison"
print "True: %6.3g \t %6.3g \t %6.3g \t %6.3g"%(mAnde[0][1],mAnde[0][2],mAnde[1][1],mAnde[1][2])
print "A,Best: %6.3g \t %6.3g \t %6.3g \t %6.3g"%(best_params[1], best_params[2], best_params[4], best_params[5])
print "N,Best: %6.3g \t %6.3g \t %6.3g \t %6.3g"%(nbFreeEcc[0], nbFreeEcc[1], nbFreeEcc[2], nbFreeEcc[3])
print
print "model \t X^2 (true) \t X^2 (best) \t m1 error \t m2 error"
print  "%s \t %.5g \t %.5g \t %.3g \t %.3g "%("A,1S",chi2true1sOnly,chi2best1sOnly, (1.e-5 -best_params[0])/1.e-5 , (1.e-5 -best_params[3])/1.e-5)
print  "%s \t %.5g \t %.5g \t %.3g \t %.3g "%("A,Full",chi2true,chi2best, (1.e-5 -best_params[0])/1.e-5 , (1.e-5 -best_params[3])/1.e-5 )
print  "%s \t %.5g \t %.5g \t %.3g \t %.3g"%("N",noiseTotal,chi2nbest, (1.e-5 -bestNbody[0])/1.e-5 , (1.e-5 - bestNbody[3])/1.e-5)
