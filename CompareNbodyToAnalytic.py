import os
who =os.popen("whoami") 
if who.readline().strip() =='samuelhadden':
	print "On laptop..."
	TTVFAST_PATH = "/Users/samuelhadden/15_TTVFast/TTVFast/c_version/myCode/PythonInterface"
	ANALYTIC_TTV_PATH = "/Users/samuelhadden/13_HighOrderTTV/TTVEmcee"
else:
	print "On Quest..."
	TTVFAST_PATH = "/projects/b1002/shadden/7_AnalyticTTV/03_TTVFast/PyTTVFast"
	ANALYTIC_TTV_PATH = "/projects/b1002/shadden/7_AnalyticTTV/01_MCMC/00_source_code"
who.close()
import glob
import sys

sys.path.append(TTVFAST_PATH)
sys.path.append(ANALYTIC_TTV_PATH)

import PyTTVFast as nTTV
import AnalyticMultiplanetFit as aTTV
import numpy as np
import matplotlib.pyplot as pl

create_data = False

# Read in parameters from 'inpars.txt' and generate transit times via n-body
#pars = np.loadtxt('inpars.txt')
pars = np.loadtxt('bestpars.txt')
if create_data:
	nbody_compute = nTTV.TTVCompute()
	trTimes,success = nbody_compute.TransitTimes(200.,pars)
	inptData = []
	for times in trTimes:
		NandT=np.vstack(( np.arange(len(times)) , times , 1.e-5*np.ones(len(times)) )).T
		inptData.append(NandT)
	noiseLvl = 0.5e-4
	noisyData = []
	noiseTotal = 0.0
	for i,times in enumerate(trTimes):
		nTransits = len(times)
		noise = np.random.normal(0.,noiseLvl,nTransits)
		noisyData.append(np.vstack(( np.arange(nTransits) , times + noise , noiseLvl*np.ones(len(times)) )).T)
		noiseTotal+= -0.5*np.sum( noise**2 / noiseLvl**2 )
else:
	with open('planets.txt') as fi:
		plfiles = [line.strip() for line in fi.readlines()]
	noisyData= [ np.loadtxt(planet) for planet in plfiles ]

# Create an analytic fit object based on the noisy N-body transit times
analyticFit = aTTV.MultiplanetAnalyticTTVSystem(noisyData)
nbodyFit = nTTV.TTVFitnessAdvanced(noisyData)

# Convert parameters to form used by analytic fit
mAnde,pAndl = analyticFit.TTVFastCoordTransform(pars)

# Generate analytic transit times for the 'true' masses and eccentricities
pAndLbest = analyticFit.bestFitPeriodAndLongitude(mAnde)
transits = analyticFit.parameterTransitTimes(pAndLbest)

# Compute the fitness of the 'true' parameters for various analytic TTV models
chi2true = analyticFit.parameterFitness(pAndLbest)
chi2true1sOnly = analyticFit.parameterFitness(pAndLbest,Only_1S=True)
chi2true_exF = analyticFit.parameterFitness(pAndLbest,exclude=['F'])
chi2true_ex2S = analyticFit.parameterFitness(pAndLbest,exclude=['2S'])

# Find the best-fit masses and eccentricities for the analytic models and determine the transit times
best_params = analyticFit.bestFitParameters(pAndLbest)[0]
bestTransits = analyticFit.parameterTransitTimes(best_params)
chi2best = analyticFit.parameterFitness(best_params)

chi2best1sOnly = analyticFit.parameterFitness(best_params,Only_1S=True)

best_ex2S=analyticFit.bestFitParameters(best_params,exclude=['2S'])[0]
chi2best_ex2S = analyticFit.parameterFitness(best_ex2S,exclude=['2S'])
Transits_ex2S = analyticFit.parameterTransitTimes(best_ex2S)

best_exF=analyticFit.bestFitParameters(best_params,exclude=['F'])[0]
chi2best_exF = analyticFit.parameterFitness(best_exF,exclude=['F'])
Transits_exF = analyticFit.parameterTransitTimes(best_exF)

# Find best-fit masses and eccentricities for the nbody model and determine the times
if create_data:
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
else:
	coplanarPars0 = pars
	noiseTotal = nbodyFit.CoplanarParametersFitness(pars)

bestNbody = nbodyFit.CoplanarParametersTTVFit(coplanarPars0)[0]
chi2nbest = nbodyFit.CoplanarParametersFitness(bestNbody)
bestNtransits = nbodyFit.CoplanarParametersTransformedTransits(bestNbody,observed_only=True)[0]
bestNtransits = [np.vstack((np.arange(len(x)),x)).T for x in bestNtransits ]

###################################################################################
# Plot N-body and analytic transit times


for i,timedata in enumerate(zip(transits,bestTransits,Transits_ex2S,Transits_exF,bestNtransits,noisyData)):
	times,bestTimes,ex2Stimes,exFtimes,bestNtimes,obstimes = timedata

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

pl.figure(3)
analyticFit.parameterTTV1SResidualsPlot(best_ex2S,exclude=['2S'],fmt = 'g.')
analyticFit.parameterTTV1SResidualsPlot(best_exF,exclude=['F'], fmt='b.')
analyticFit.parameterTTV1SResidualsPlot(np.array(( best_params,pAndLbest )),exclude=[],fmt = 'k.')
pl.show()
nbEx = bestNbody[2:3*analyticFit.nPlanets:3]
nbEy = -bestNbody[1:3*analyticFit.nPlanets:3]
nbMandE = np.hstack( (  bestNbody[:3*analyticFit.nPlanets:3].reshape(-1,1) , vstack((nbEx,nbEy)).T ) )
#nbFreeEcc = analyticFit.forcedEccs(np.array((bestNbody[0],bestNbody[2],-bestNbody[1], bestNbody[3], bestNbody[5], -bestNbody[4])),pAndl)[1].reshape(-1)
nbFreeEcc = analyticFit.forcedEccs(nbMandE,pAndl)[1].reshape(-1)
nPlanets = analyticFit.nPlanets
print "Free eccentricity comparison"
print "  True: ","\t".join(map(lambda x: "%+5.3f"%x , mAnde[:,(1,2)].reshape(-1) ) )
best_eccs = best_params[:analyticFit.nPlanets*3].reshape(-1,3)[:,(1,2)].reshape(-1)
print "A,Best: ","\t".join(map(lambda x: "%+5.3f"%x , best_eccs ) )
print "N,Best: ","\t".join(map(lambda x: "%+5.3f"%x , nbFreeEcc ) )

print

print "model \t X^2 (true) \t X^2 (best) \t dm/m"

MassErr = (pars[::3][:nPlanets] - best_params[::3][:nPlanets] )/ (pars[::3][:nPlanets])
print  "%s \t %.5g \t %.5g \t"%("A,1S",chi2true1sOnly,chi2best1sOnly),
print	"\t".join( map(lambda x: "%+5.3g"%x , MassErr ) )

MassErr = (pars[::3][:nPlanets] - best_exF[::3][:nPlanets] )/ (pars[::3][:nPlanets])
print  "%s \t %.5g \t %.5g \t"%("A,1/2S",chi2true_exF, chi2best_exF),
print	"\t".join( map(lambda x: "%+5.3g"%x , MassErr ) )


MassErr = (pars[::3][:nPlanets] - best_ex2S[::3][:nPlanets] )/ (pars[::3][:nPlanets])
print  "%s \t %.5g \t %.5g \t"%("A,1S+F",chi2true_ex2S,chi2best_ex2S),
print	"\t".join( map(lambda x: "%+5.3g"%x , MassErr ) )

MassErr = (pars[::3][:nPlanets] - best_params[::3][:nPlanets] )/ (pars[::3][:nPlanets])
print  "%s \t %.5g \t %.5g \t"%("A,Full",chi2true,chi2best),
print	"\t".join( map(lambda x: "%+5.3g"%x , MassErr ) )

MassErr = (pars[::3][:nPlanets] - bestNbody[::3][:nPlanets] )/ (pars[::3][:nPlanets])
print  "%s \t %.5g \t %.5g \t"%("N",noiseTotal,chi2nbest),
print	"\t".join( map(lambda x: "%+5.3g"%x , MassErr ) )
