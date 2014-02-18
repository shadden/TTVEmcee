from ttvfuncs import *
import numpy as np
import matplotlib.pyplot as pl
#------------------------------------------
#     Parameter Plot
#------------------------------------------
def myhist2d(xdata,ydata,xlims=-99,ylims=-99,nxbins=25,nybins=25):
	x = xdata
	y = ydata
	if(xlims==-99): xlims=[min(x),max(x)]
	if(ylims==-99): ylims=[min(y),max(y)]
#
	xmin = min(xlims)
	xmax = max(xlims)
	ymin = min(ylims)
	ymax = max(ylims)
#
	xbins = np.linspace( start = xmin, stop = xmax, num = nxbins )
	ybins = np.linspace( start = ymin, stop = ymax, num = nybins )
#
	H,xedges,yedges = np.histogram2d(y,x,normed=True,bins=(ybins,xbins))
	pl.imshow(H,extent=[xmin,xmax,ymin,ymax],interpolation='nearest',origin='lower',aspect=(xmax/ymax) )
def myhist2d1d(xdata,ydata,xlims=-99,ylims=-99,nxbins=25,nybins=25):
	x = xdata
	y = ydata
	if(xlims==-99): xlims=[min(x),max(x)]
	if(ylims==-99): ylims=[min(y),max(y)]
#
	xmin = min(xlims)
	xmax = max(xlims)
	ymin = min(ylims)
	ymax = max(ylims)
#
	xbins = np.linspace( start = xmin, stop = xmax, num = nxbins )
	ybins = np.linspace( start = ymin, stop = ymax, num = nybins )
#
	H,xedges,yedges = np.histogram2d(y,x,normed=True,bins=(ybins,xbins))
	pl.imshow(H,extent=[xmin,xmax,ymin,ymax],interpolation='nearest',origin='lower',aspect=(xmax/ymax) )
def par_plot(pars):
      #--------------------------------------
      n,t,err = input_data.T
      n1,t1,err1 = input_data1.T
      t0,t01,p,p1,m,m1,ex,ey,ex1,ey1 = pars
      Vx,Vy,Vx1,Vy1=get_ttv_coefficients(m,m1,ex,ey,ex1,ey1,p,p1)
      V,argV=sqrt(Vx*Vx+Vy*Vy),arctan2(Vy,Vx)
      V1,argV1=sqrt(Vx1*Vx1+Vy1*Vy1),arctan2(Vy1,Vx1)
      dtcal,dtcal1=[],[]
      t0,t01,p,p1,m,m1,ex,ey,ex1,ey1 = pars
      pRatio = p/p1
      deltas = array([ (j-1.)/j / pRatio - 1. for j in arange(2,6) ])
      j, DD = min(enumerate(abs(deltas)), key=lambda x: x[1])
      j=j+2
      for point in input_data:
            n,tt,err = point
            lj = 2*pi * ( j* (tt-t01)/p1 - (j-1.) * (tt-t0)/p)
            dtcal.append( V * sin(lj + argV) )
      for point in input_data1:
            n,tt,err = point
            lj = 2*pi * ( j* (tt-t01)/p1 - (j-1.) * (tt-t0)/p)
            dtcal1.append( V1 * sin(lj + argV1) )
      dtcal,dtcal1=array([dtcal,dtcal1])
      pl.figure()
      pl.plot(t,dtcal,'bx-')
      pl.plot(t1,dtcal1,'gx-')
      n,t,err = input_data.T
      n1,t1,err1 = input_data1.T
      pl.errorbar(t,t-t0-p*n,yerr=err,fmt='bo')
      pl.errorbar(t1,t1-t01-p1*n1,yerr=err1,fmt='go')
      pl.show()
#------------------------------------------
#     Chain to TTV Coefficients
#------------------------------------------
def chain2TTVcoeff(chain):
	return array([get_ttv_coefficients(x[4],x[5],x[6],x[7],x[8],x[9],x[2],x[3]) for x in chain])
#------------------------------------------
#     Chain to TTV Coefficients
#------------------------------------------
def chain2Z(chain):
	p,p1=chain.T[2:4]
	ex,ey,ex1,ey1 = chain.T[6:]
	pRatio = p/p1
	j,alpha,delta,F,F1,F1s = array([ laplace_coefficients(pr) for pr in pRatio]).T
	Zx = F * ex + F1 * ex1
	Zy = F * ey + F1 * ey1
	return array([Zx,Zy]).T
