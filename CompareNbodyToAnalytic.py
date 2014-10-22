
import glob
import sys

sys.path.append("/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface")
sys.path.append("/Users/samuelhadden/13_HighOrderTTV/TTVEmcee/")

import PyTTVFast as nTTV
import AnalyticMultiplanetFit as aTTV

# Read in parameters from 'inpars.txt' and generate transit times via n-body
pars = np.loadtxt('inpars.txt')
nbody_compute = nTTV.TTVCompute()
trTimes,success = nbody_compute.TransitTimes(200.,pars)

inptData = []

for times in trTimes:
	NandT=np.vstack(( np.arange(len(times)) , times , 1.e-5*np.ones(len(times)) )).T
	inptData.append(NandT)

# Create an analytic fit object based on the N-body transit times
analyticFit = aTTV.MultiplanetAnalyticTTVSystem(inptData)


# Convert parameters to form used by analytic fit


mAnde,pAndl = analyticFit.TTVFastCoordTransform(pars)
# Generate analytic transit times from full parameter list

pAndLbest = analyticFit.bestFitPeriodAndLongitude(mAnde)
transits = analyticFit.parameterTransitTimes(pAndLbest)

new_params=np.hstack((massAndEccs.reshape(-1),transformedPersAndLs.reshape(-1)[2:]))
best_params = analyticFit.bestFitParameters(new_params)[0]
bestTransits = analyticFit.parameterTransitTimes(best_params)


noiseLvl = 2.e-4
pl.figure(1)
for i,times in enumerate(trTimes):
	nTransits = len(times)
	noise = np.random.normal(0.,noiseLvl,nTransits)
	noisyData=np.vstack(( np.arange(nTransits) , times + noise , noiseLvl*np.ones(len(times)) )).T
	np.savetxt("planet%d.txt"%i,noisyData)
	pl.errorbar(noisyData[:,1],linefit_resids(noisyData[:,0],noisyData[:,1],noisyData[:,2]),yerr=noisyData[:,2],fmt='s')
	

		
# Plot N-body and analytic transit times

for i,timedata in enumerate(zip(transits,bestTransits,inptData)):
	times,bestTimes,obstimes = timedata

	# Plot the TTVs
	pl.figure(1)	
	pl.subplot(analyticFit.nPlanets*100+10 + i + 1)
	
	ttvs = linefit_resids(times[:,0],times[:,1])
	bestTtvs = linefit_resids(bestTimes[:,0],bestTimes[:,1])
	
	obs_ttvs = linefit_resids(obstimes[:,0],obstimes[:,1])
	pl.errorbar(obstimes[:,1],obs_ttvs,yerr=obstimes[:,2],fmt='rs')
	
	pl.plot(times[:,1],ttvs,'k.')
	pl.plot(bestTimes[:,1],bestTtvs,'g.')
	
	
	# Plot the TTV residuals	
	pl.figure(2)	
	pl.subplot(analyticFit.nPlanets*100+10 + i + 1)
	resids = obstimes[:,1] - times[:,1]
	bestResids = obstimes[:,1] - bestTimes[:,1]
	
	pl.errorbar(obstimes[:,1],resids,yerr=obstimes[:,2],fmt='rs')
	pl.errorbar(obstimes[:,1],bestResids,yerr=obstimes[:,2],fmt='gs')

pl.show()

