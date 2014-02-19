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
	assert type(j) == int, "Laplace coefficient must have integer j value"
	#assert type(p) == int, "Laplace coefficient must have integer p value"
	integrand = lambda phi: cos(j * phi ) / ( 1. - 2. * alpha * cos(phi) +alpha**2 )**s
	result=integ.quad(integrand ,0,2*pi) 
	return result[0]/ ( pi)

def b(alpha,s,j,p):
	"""The p-th derivative Laplace coefficient, b_s^(j)[alpha]"""
	assert type(p) == int, "Laplace coefficient must have integer p value"
	return deriv(lambda x:b0(x,s,j),alpha,p)
def f(alpha,j):
	return -j*b(alpha,0.5,j,0) - 0.5 * alpha * b(alpha,0.5,j,1) 
def f1(alpha,j,perturber="External"):
	direct = (-0.5+j)*b(alpha,0.5,j-1,0) + 0.5 * alpha * b(alpha,0.5,j-1,1)
	if j==2: 
		if perturber=="External":
			return direct - 2.*alpha
		elif perturber =="Internal":
			return direct - 0.5*alpha**(-2.)
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
def k1(alpha,j):
	direct =  0.5 * b(alpha,0.5,j,0)
	if j==1:
		return direct - 0.5 / (alpha**2)
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
		perturber=="Internal":
			return term1 + term2 + term3 - 3. *0.125 /( alpha**2 )
		elif perturber=="External":
			return term1 + term2 + term3
		else:
			raise Exception("Invalid perturber type!")
	else:
		return term1 + term2 + term3
def h(alpha,j):
	term1 = (-0.5 + 1.5*j - j**2)*b(alpha,0.5,j-1,0)
	term2 = (0.5 - j) * alpha *b(alpha,0.5,j-1,1)
	term3 = 0.25*alpha**2 * b(alpha,0.5,j-1,2)

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
def get_k_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ k(alpha,j) for j in arange(1,5)])
def get_k1_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ k1(alpha,j) for j in arange(1,5)])
def get_g_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ g(alpha,j) for j in arange(3,9)])
def get_g1Ext_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ g(alpha,j,perturber="External") for j in arange(3,9)])
def get_g1Int_array(pratio):
	alpha = pratio**(-2./3.)
	return array( [ g(alpha,j,perturber="Internal") for j in arange(3,9)])
def get_h_array(pratio,j):
	alpha = pratio**(-2/3.)
	return array( [ h(alpha,j) for j in arange(3,9)])
#########################################################
# old arrays
#########################################################

f =array( \
	[[-1.19049, -2.17528, -3.01071, -3.76251],\
 	[-0.925781, -2.02522,-3.00141, -3.8814],\
 	[-0.676302, -1.78148, -2.84043, -3.81606], \
	[-0.477612, -1.51855, -2.61021, -3.64962]] )
f1Int =array( \
	[[0.42839, 1.70256, 2.56064, 3.28928],\
	[1.37315, 2.48401, 3.40977,4.23171],\
  	[1.02274, 2.22548, 3.28326, 4.22917], \
  	[0.730117, 1.91689, 3.04747, 4.08371]] )

f1Ext =array( \
	[[0.42839, 1.03481, 1.64344, 2.23899],\
	 [1.37315, 2.48401, 3.40977, 4.23171], \
	[1.02274, 2.22548, 3.28326, 4.22917], \
	[0.730117, 1.91689, 3.04747, 4.08371]] )

df = array(\
	[[-5.23115, -10.5482, -17.1517, -25.1309],\
 	[-5.48165, -12.2139,-20.1177, -29.3127],\
 	[-5.05767, -12.9583, -22.1618, -32.6315], \
 	[-4.32141, -12.973, -23.3441, -35.0656]])
df1Int = array(\
	[[8.78495, 11.4805, 16.9164, 24.0553],\
 	[5.96114, 11.7805, 18.8652, 27.3191],\
 	[6.02774, 13.2932, 21.705, 31.3953],\
 	[5.44783, 13.8734, 23.5963, 34.5751]])
df1Ext = array(\
	[[2.78495, 7.2305, 13.1386, 20.4928],\
	 [5.96114, 11.7805, 18.8652,27.3191],\
	 [6.02774, 13.2932, 21.705, 31.3953],\
	 [5.44783, 13.8734, 23.5963, 34.5751]])

#########################################################
# one-to-one terms
#########################################################
kj=array([[0.0634399, 0.138193, 0.201314, 0.254575],\
	 [0.182657, 0.308725, 0.399064, 0.469205],\
	 [0.0969614, 0.200258, 0.281848, 0.347772],\
	 [0.0538151, 0.1354, 0.207036, 0.267644],\
	 [0.0306544, 0.0938033, 0.155639, 0.210596]])

k1j=array([[-0.881501, -0.338772, -0.119707, 0.0122013],\
	 [0.182657, 0.308725, 0.399064, 0.469205],\
	 [0.0969614, 0.200258, 0.281848, 0.347772],\
	 [0.0538151, 0.1354, 0.207036, 0.267644],\
	 [0.0306544, 0.0938033, 0.155639, 0.210596]])

dkj=array([[0.377913, 0.812738, 1.25926, 1.71189],\
	 [0.72999, 1.23225, 1.71348, 2.18815],\
	 [0.546087, 1.07932, 1.58734, 2.08264],\
	 [0.390154, 0.915007, 1.43449, 1.94356],\
	 [0.271553, 0.760698, 1.27661, 1.79125]])

dk1j=array([[4.87791, 3.56274, 3.53704, 3.77439],\
	 [0.72999, 1.23225, 1.71348, 2.18815],\
	 [0.546087, 1.07932, 1.58734, 2.08264],\
	 [0.390154, 0.915007, 1.43449, 1.94356],\
	 [0.271553, 0.760698, 1.27661, 1.79125]])
#########################################################
# second Order Terms 
#########################################################
g = array([[1.63036, 3.99935, 6.61413, 9.48711],\
	 [1.69573, 4.83452, 8.33111, 12.068],\
	 [1.55472, 5.24979, 9.57994, 14.2047],\
	 [1.32207, 5.33217, 10.3828, 15.8659],\
	 [1.06933, 5.17442, 10.7964, 17.0677],\
	 [0.834549, 4.85795, 10.8887, 17.8502]])

g1Ext=array([[1.06533, 2.89715, 5.12626, 7.70988],\
	 [3.59379, 7.06119, 10.5132, 14.1145],\
	 [3.43345, 7.94967, 12.476, 17.076],\
	 [2.99452, 8.26209, 13.8049, 19.4324],\
	 [2.46396, 8.14619, 14.5669, 21.1881],\
	 [1.94696, 7.73761, 14.8524, 22.386]])

g1Int=array([[2.2465, 4.82885, 7.36193, 10.1134],\
	 [3.59379, 7.06119, 10.5132, 14.1145],\
	 [3.43345, 7.94967, 12.476, 17.076],\
	 [2.99452, 8.26209, 13.8049, 19.4324],\
	 [2.46396, 8.14619, 14.5669, 21.1881],\
	 [1.94696, 7.73761, 14.8524, 22.386]])

h=array([[-4.62394, -9.46309, -14.606, -20.2348],\
	 [-4.96685, -11.7488, -18.8083, -26.2154],\
	 [-4.63643, -12.9612, -21.929, -31.2328],\
	 [-3.98802, -13.3022, -23.9912, -35.1822],\
	 [-3.25127, -13.0037, -25.1164, -38.0837],\
	 [-2.55218, -12.2751, -25.4605, -40.0198]])
