from numpy import *
from scipy import integrate as integ
##
#	Laplace coefficients:
#		f[n-2,m-2] = f_n[ ((m-1)/m)**2/3 ]
##
#########################################################
# numerical calculations of coefficients
#########################################################
def deriv(fn,x,n,eps=1.e-5):
	assert type(n)==int,"Must give integer argument for derivative!"
	if n==0:
		return fn(x)
	else:
		dfn = lambda z: deriv(fn,z,n-1,eps=eps)
		return (dfn(x+0.5*eps) - dfn(x-0.5*eps)) / eps
def b0(alpha,s,j):
	"""Laplace coefficient, b_s^(j)[alpha]"""
	assert type(j) == int or type(j) == int64, "Laplace coefficient must have integer j value, %s was given" % type(j)
	integrand = lambda phi: cos(j * phi ) / ( 1. - 2. * alpha * cos(phi) +alpha**2 )**s
	result=integ.quad(integrand ,0,2*pi) 
	return result[0]/ ( pi)

def b(alpha,s,j,p):
	"""The p-th derivative Laplace coefficient, b_s^(j)[alpha]"""
	assert type(p) == int, "Laplace coefficient must have integer p value"
	return deriv(lambda x:b0(x,s,j),alpha,p)
def f(alpha,j):
	return -j*b(alpha,0.5,j,0) - 0.5 * alpha * b(alpha,0.5,j,1) 
def df(alpha,j):
	return -j*b(alpha,0.5,j,1) - 0.5 * b(alpha,0.5,j,1) - 0.5 * alpha * b(alpha,0.5,j,2)
def f1(alpha,j,perturber="External"):
	direct = (-0.5+j)*b(alpha,0.5,j-1,0) + 0.5 * alpha * b(alpha,0.5,j-1,1)
	if j==2: 
		if perturber=="External":
			return direct - 2.* alpha
		elif perturber == "Internal":
			return direct - 0.5*alpha**(-2.)
		else:
			raise Exception("Invalid perturber type!")
	else:
		return direct
def df1(alpha,j,perturber="External"):
	direct = (-0.5+j)*b(alpha,0.5,j-1,1) + 0.5 * alpha * b(alpha,0.5,j-1,2) +0.5 * b(alpha,0.5,j-1,1)
	if j==2: 
		if perturber=="External":
			return direct - 2.
		elif perturber == "Internal":
			return direct + alpha**(-3.)
		else:
			raise Exception("Invalid perturber type!")
	else:
		return direct
def k(alpha,j):
	direct = 0.5 * b(alpha,0.5,j,0)
	if j==1:
		return direct - 0.5 * alpha
	else:
		return direct
def dk(alpha,j):
	direct = 0.5 * b(alpha,0.5,j,1)
	if j==1:
		return direct - 0.5 
	else:
		return direct
def k1(alpha,j):
	direct =  0.5 * b(alpha,0.5,j,0)
	if j==1:
		return direct - 0.5 / (alpha**2)
	else: return direct
def dk1(alpha,j):
	direct =  0.5 * b(alpha,0.5,j,1)
	if j==1:
		return direct + 1.0 / (alpha**3)
	else:
		return direct
def g(alpha,j):
	term1 = (0.5*j**2 - j*5./8.) * b(alpha,0.5,j,0)
	term2 = (-0.25 + 0.5 * j) * alpha * b(alpha,0.5,j,1) 
	term3 =  0.125 * alpha**2 * b(alpha,0.5,j,2)
	return term1 + term2 + term3
def g1(alpha,j,perturber="External"):
	term1 = (0.25 -7.*j/8. + j**2 /2.) * b(alpha,0.5,j-2,0)
	term2 = (0.5*j-0.25) * alpha * b(alpha,0.5,j-2,1)
	term3 = 0.125 * alpha**2 * b(alpha,0.5,2-j,2)
	if j==3:
		if perturber=="Internal":
			return term1 + term2 + term3 - 3. *0.125 /( alpha**2 )
		elif perturber=="External":
			return term1 + term2 + term3 - 27. * 0.125 * alpha
		else:
			raise Exception("Invalid perturber type!")
	else:
		return term1 + term2 + term3
def h(alpha,j):
	term1 = (-0.5 + 1.5*j - j**2)*b(alpha,0.5,j-1,0)
	term2 = (0.5 - j) * alpha *b(alpha,0.5,j-1,1)
	term3 = -0.25*alpha**2 * b(alpha,0.5,j-1,2)
	return term1 + term2 + term3

#########################################################
# array-returning functions
#########################################################
def get_f_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ f(alpha,j) for j in arange(2,6)])
def get_f1Int_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ f1(alpha,j,perturber="Internal") for j in arange(2,6)])
def get_f1Ext_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ f1(alpha,j,perturber="External") for j in arange(2,6)])
def get_df_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ df(alpha,j) for j in arange(2,6)])
def get_df1Int_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ df1(alpha,j,perturber="Internal") for j in arange(2,6)])
def get_df1Ext_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ df1(alpha,j,perturber="External") for j in arange(2,6)])
def get_k_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ k(alpha,j) for j in arange(1,6)])
def get_k1_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ k1(alpha,j) for j in arange(1,6)])
def get_dk_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ dk(alpha,j) for j in arange(1,6)])
def get_dk1_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ dk1(alpha,j) for j in arange(1,6)])
def get_g_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ g(alpha,j) for j in arange(3,9)])
def get_g1Ext_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ g1(alpha,j,perturber="External") for j in arange(3,9)])
def get_g1Int_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ g1(alpha,j,perturber="Internal") for j in arange(3,9)])
def get_h_array(pratio):
	alpha = pratio**(-2/3.)
	return array( [ h(alpha,j) for j in arange(3,9)])
