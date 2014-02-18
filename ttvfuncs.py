from numpy import *
#------------------------------------------
#     MCMC PARAMETERS
#------------------------------------------
#     0,1 : t0,t01
#     2,3 : p,p1
#     4,5 : m,m1
#     6,7 : ex,ey
#     8,9 : ex1,ey1
#------------------------------------------
#     Efficiently determine initial parameter settings
#------------------------------------------
def linearfit(x,y):
      assert len(x)==len(y), "length of arrays unequal! x: {}, y: {}".format(len(x),len(y))
      a = array([ones(len(x)),x]).T
      return linalg.lstsq(a,y)[0]
def initial_parameters(transits,transits1):
      # transits should take the form "n,t"
      n,t=transits.T
      n1,t1 = transits1.T
      t0,p = linearfit(n,t)
      t01,p1= linearfit(n1,t1)
      pRatio = p/p1
      deltas = array([ (j-1.)/j / pRatio - 1. for j in arange(2,6) ])
      j, DD = min(enumerate(abs(deltas)), key=lambda x: x[1])
      j=j+2.
      w = 2*pi * (j/p1 - (j-1.)/p)
      lj0 = 2*pi * (j*(-t01)/p1 - (j-1)*(-t0)/p)
      a = array([ones(len(n)), n , sin(w*t+lj0) ,cos(w*t+lj0)]).T
      t0,p,a,b=linalg.lstsq(a,t)[0]
      a1 = array([ones( len(n1)), n1 , sin(w*t1+lj0) ,cos(w*t1+lj0)]).T
      t01,p1,a1,b1=linalg.lstsq(a1,t1)[0]
      mnom = sqrt(a1*a1)*DD*pi*j/p1
      m1nom = sqrt(a*a) * DD*pi * j**(2./3.) *(j-1.)**(1./3.) / p
      ey = 0. #-1.*DD**2/m1nom * 1.5 * pi * j / p 
      print "Initial parameters determined to be:\n"
      print "j: {}\n".format(j)
      print "T0: {:6.3f}\n".format(t0)
      print "T01: {:6.3f}\n".format(t01)
      print "P: {:6.3f}\n".format(p)
      print "P1: {:6.3f}\n".format(p1)
      print "m: {:.3g}\n".format(mnom)
      print "m1: {:.3g}\n".format(m1nom)
      return t0,t01,p,p1,mnom,m1nom,0,ey,0,0
#------------------------------------------
#     initialize walkers
#------------------------------------------
def initialize_walkers(nwalkers,par0):
      t0,t01,p,p1,m,m1,ex,ey,ex1,ey1 = par0
      p = random.normal(p,.01*p,size=nwalkers)
      p1 = random.normal(p1,.01*p1,size=nwalkers)
      t0 = random.normal(t0,.1*t0,size=nwalkers)
      t01 = random.normal(t01,.1*t01,size=nwalkers)
      m = random.lognormal(log(m),0.1,size=nwalkers)
      m1 = random.lognormal(log(m1),0.1,size=nwalkers)
      ex,ey,ex1,ey1 = random.normal(scale=0.2,size=(4,nwalkers))
      return array([t0,t01,p,p1,m,m1,ex,ey,ex1,ey1]).T
#-------------------------------------------
#  convert mass,eccentricity to ttv coefficients
#-------------------------------------------
def get_ttv_coefficients(m,m1,ex,ey,ex1,ey1,p,p1):
      j,alpha,Delta,fj,fj1,fj1s = laplace_coefficients(p/p1)
      Zx,Zy = fj*ex+fj1*ex1, fj*ey+fj1*ey1
      Zxs,Zys = fj*ex+fj1s*ex1, fj*ey+fj1s*ey1
      A= p / ( pi * sqrt(alpha) *j )
      B = p1 / (pi * j)
      C = ((j-1.) / j) * alpha**(-1.5)
      #-----------------------------------------------#
      Vx = A * (m1/Delta) * (-fj - 1.5 * C * Zx / Delta)
      Vx1 = B * (m/Delta) * ( -fj1s + 1.5 * Zxs / Delta)
      Vy = A * (m1/Delta) * ( + 1.5 *C* Zy / Delta)
      Vy1 = B * (m/Delta) * ( - 1.5 * Zys / Delta)
      #----------------------------------------------#
      return array([Vx,Vy,Vx1,Vy1])
#------------------------------------------
#  Priors
#------------------------------------------
def logp(pars):
      m,m1 = pars[4:6]
      if m < 0. or m1 < 0.:
            return -inf
      ex,ey,ex1,ey1 = pars[6:]
      if ex*ex+ey*ey > 0.8 or ex1*ex1+ey1*ey1 > 0.8:
            return -inf
      sigma_e = 0.012
      return -(ex**2+ey**2+ex1**2+ey1**2) / (2.*sigma_e**2)  - 4.*log( sqrt(2*pi*sigma_e**2) )
#-------------------------------------------
#  Laplace coefficients
#-------------------------------------------
def laplace_coefficients(pRatio):
      deltas = array([ (j-1.)/j / pRatio - 1. for j in arange(2,6) ])
      j, DD = min(enumerate(abs(deltas)), key=lambda x: x[1])
      DD = deltas[j]
      alpha = pRatio**(2./3.)
      j=j+2
      if j==2:
            F=-1.1904936978495035 + 2.1969462769618904*DD - 3.8289055653018256*DD**2 + 6.771009771171885*DD**3
            F1=0.428389834143899 - 1.1696066294495702*DD + 2.5280781617318087*DD**2 - 5.091740584580391*DD**3
            F1s = 0.42838983414389875 - 3.689448729239315*DD + 2.9480518450300974*DD**2 - 5.651705495644773*DD**3
      if j==3:
            F = -2.025222689938595 + 6.213974290251607*DD - 16.304376098697116*DD**2 + 42.827419060576965*DD**3
            F1= 2.4840051833039443 - 5.993444597443975*DD + 14.796224048738608*DD**2 - 38.866311515357346*DD**3
      if j==4:
            F=-2.84043 + 12.1961*DD - 42.6232*DD**2 + 148.355*DD**3
            F1=3.283256721822216 - 11.94471792376407*DD + 39.79304293074331*DD**2 - 138.53068381843542*DD**3
      if j==5:
            F=-2.6102063540681852 + 12.846737150348432*DD - 47.23760385628128*DD**2 + 165.1299299321432*DD**3
            F1 = 3.047468173636637 - 12.985522003237872*DD + 44.97818216055746*DD**2 - 155.21028765265908*DD**3
      if j!= 2:
            F1s = F1
      return array([j,alpha,DD,F,F1,F1s])
            
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

