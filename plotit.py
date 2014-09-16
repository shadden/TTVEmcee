import sys
sys.path.insert(0, '/projects/b1002/shadden/7_AnalyticTTV/01_MCMC/00_source_code')
import fitnessNEW as fff
import triangle

data = loadtxt('chain.00.dat.gz')
data2 = array([log10(data[:,0]),log10(data[:,1]),data[:,2],data[:,3],data[:,4],data[:,5]]).T
lnlike = loadtxt('chain.00.lnlike.dat.gz').reshape(-1,)
fCoeff = -2.02522
f1Coeff = 2.48401
m,m1,ex,ex1,ey,ey1 = data.T
bestpars = data[argmax(lnlike.reshape((-1,)))]
truepars = loadtxt("TrueParams.txt")
Zx = fCoeff * ex + f1Coeff * ex1
Zy = fCoeff * ey + f1Coeff * ey1
absZ = sqrt(Zx**2 + Zy **2)

#pts = array([ ( 10**logm, median(absZ[m>10**logm]) ) for logm in linspace(min(log10(m)),max(log10(m)),30) if len(absZ[m>10**logm]) !=0]  )
#pts1 = array([ ( 10**logm1, median(absZ[m1>10**logm1])) for logm1 in linspace(min(log10(m1)),max(log10(m1)),30) if len(absZ[m1>10**logm1]) !=0 ])
#
#pts_75 = array([ ( 10**logm, percentile(absZ[m>10**logm],75) ) for logm in linspace(min(log10(m)),max(log10(m)),30) if len(absZ[m>10**logm]) !=0 ])
#pts1_75 = array([ ( 10**logm1, percentile(absZ[m1>10**logm1],75)) for logm1 in linspace(min(log10(m1)),max(log10(m1)),30) if len(absZ[m1>10**logm1]) !=0] )
#
#pts_25 = array([ ( 10**logm, percentile(absZ[m>10**logm],25) ) for logm in linspace(min(log10(m)),max(log10(m)),30) if len(absZ[m>10**logm]) !=0 ])
#pts1_25 = array([ ( 10**logm1, percentile(absZ[m1>10**logm1],25)) for logm1 in linspace(min(log10(m1)),max(log10(m1)),30) if len(absZ[m1>10**logm1]) !=0])

#figure()
#semilogx(pts[:,0],pts[:,1],'k-',label='inner planet, median')
#semilogx(pts_25[:,0],pts_25[:,1],'k--')
#semilogx(pts_75[:,0],pts_75[:,1],'k--')
#semilogx(pts1[:,0],pts1[:,1],'r-',label='outer planet, median')
#semilogx(pts1_25[:,0],pts1_25[:,1],'r--')
#semilogx(pts1_75[:,0],pts1_75[:,1],'r--')
#xmin = min( [min(m) ,min(m1) ] )
#xmax = max( [max(m) ,max(m1) ] )
#xlim(xmin,xmax)
#show()
#savefig('median_ecc_versus_mass')

elims = (-0.2,0.2)
extnts = [1.,1.,elims,elims,elims,elims]
true_vals = truepars.copy()
true_vals[:2] = log10(true_vals[:2])
lbls = ['logM','logM1','ex','ey','ex1','ey1']
triangle.corner(data2,extents=extnts,labels=lbls,truths=true_vals)
show()
savefig('parameters_corner.png')

figure()
for walker in lnlike.reshape(-1,200).T:
	plot(walker)
show()
savefig('chains_plot.png')


input_data=loadtxt("./inner.ttv")
input_data1 = loadtxt("./outer.ttv")
ft = fff.fitness(input_data,input_data1)

ft.fitplot(bestpars)
subplot(211)
title("best params, lnlike: %.3f" % ft.fitness2(bestpars))
savefig('best_fit.png')

ft.fitplot(truepars)
subplot(211)
title("true params, lnlike: %.3f" % ft.fitness2(truepars))
savefig('true_fit.png')
