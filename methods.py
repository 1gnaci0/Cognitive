
###############
### IMPORTS ###
###############

import numpy as np
from scipy.optimize import fsolve
from scipy import integrate
from matplotlib import pyplot as plt
import pickle

###############
### CLASSES ###
###############

class Individual():
    """
    Class defining the object of study. It is initiated with a list (or array) with 
    the following structure: [#people in layer 1, ...,#people in layer r]
    """
    
    def __init__(self,l):
        self.l = l              # array/list [#people in layer 1, ...,#people in layer r]
        self.r = len(l)         # integer
        self.tol = 1e-6         # tolerance for fsolve()
    
    ### AUXILIARY METHODS ###

    def L1(self):
	# See SI Eq (32)
        k = np.array(range(self.r))  # 0 --> (r-1)
        return float(np.dot(self.l,k))
    def L2(self):
	# See SI Eq (32)    
        return self.L()*(float(self.r) - 1) - self.L1()
    def L(self):
	# See SI Eq (32)    
        return np.sum(self.l)  
       
    ### INTEGRATION METHODS ### 
    
    def F_t(self,t,R):
        # R is either self.L2() or self.L1(). See SI Eq(30)
        L = self.L()
        r= float(self.r)
   
        if t != np.inf:
            #For finite limits of integration use Gauss-Legendre
            return integrate.quad(integrand_leg,args = (R,L,r), a=0., b=t)[0]
         
        else:
            #For infinite limits of integration use Gauss-Laguerre
            ti, wi, = np.polynomial.laguerre.laggauss(150) 
            f_xi = [integrand_lag(t,R,L,r) for t in ti]      
            return np.dot(f_xi, wi)
        
    def G(self, t):
	# Cumulative distribution of the parameter. See SI Eq (37)
        L1 = self.L1()
        L2 = self.L2()
        F_inf_L1 = self.F_t(np.inf, L1)
        F_inf_L2 = self.F_t(np.inf, L2)
        norm = F_inf_L1 + F_inf_L2
            
        if t < 0.:
            numerator = F_inf_L1 - self.F_t(-t,L1) 
        else:
            numerator = F_inf_L1 + self.F_t(t,L2)
        
        result = numerator / norm  
        
        return result
        
    ### ESTIMATING PARAMETER METHODS ###

    def f(self, mu):
        # See SI Eq (20)    
        r = float(self.r)
        if abs(mu) < 1e-4:
            # Defined to be continuous in mu = 0 
            # First order Taylor series at 0 
            return 0.5 + (1/12.)*(r**2 - 1)*mu/(r-1) 
        else:        
            s = np.exp(mu)
            F = s*((((r-1.)*s**r)-(r*s**(r-1.))+1.) / ((r-1.)*(((s**r - 1.))*(s - 1.))))                 
            return F

    def f2fit(self, mu):
	# See SI Eq (36)     
        r = float(self.r)
        return (r-1.)*self.f(mu) - (self.L1() / self.L())
    
    def mu(self):
	# Maximum Likelihood estimation of mu    
	# Finds the solution of f2fit()
	
	mu_0 = (self.r-1.)*0.5 - (self.L1() / self.L()) # initial guess (0.5 =  mean value of f(), see SI Eq (21))       
	ans = fsolve(self.f2fit, mu_0, xtol=self.tol)[0]
	return ans
    ### MAIN METHOD: fit_model ###
    def fit_model(self, d=0.05, eps = 1e-1):
	"""
	Main Method, returns the fitted parameter and the limits of the Confidence Interval.
	(1-d) is the level of Confidence. d=0.05 --> 95%
	
	Returns: m,t1,t2
	
		m:  estimation of the parameter mu
		t1: lower limit of the confidence interval
		t2: upper limit of the confidence interval
	"""
        m = self.mu()
        initial = m
	t1 = fsolve(lambda x: self.G(x) - d/2, x0=initial - eps, xtol=self.tol)[0]     # Looks for t1 starting eps to the left of the mode
	t2 = fsolve(lambda x: self.G(x) - (1-d/2),x0=initial + eps, xtol=self.tol)[0]  # Looks for t2 starting eps to the right of the mode
	return m,t1,t2
                
#################    
### FUNCTIONS ###
#################

def integrand_leg(x,R,L,r):
    # Integrand for Gauss_Legendre quadrature.  See SI Eq (30) and Methods
    if abs(x) < 0:
	# Taylor expansion of quotient up to third order at 0
	# It prevents bad behaviours near zero (not needed for the cases analysed in the paper)
	ev_log = L*np.log((1. - x/2. + (x**2 / 6.))/r*(1. - (x*r / 2.) + ((x*r)**2 / 6.))) -x*R
	return np.exp(ev_log)
    else:	
	# First evaluate log, then exponentiate
	ev_log1 = -x*R + L*np.log((1. - np.exp(-x))/(1. - np.exp(-x*r)))
	return np.exp(ev_log1)
   
          
def integrand_lag(x,R,L,r):
    # Integrand for Gauss_Laguerre quadrature.  See SI Eq (30) and Methods
    z = x / R
    if abs(z) < 0:    
        # Taylor expansion of quotient up to third order at x=0
        # It prevents bad behaviours near zero (not needed for the cases analysed in the paper)
        return ((1. - z/2. + (z**2 / 6.))/r*(1. - (z*r / 2.) + ((z*r)**2 / 6.)))**L            
    else:
        # First evaluate log, then exponentiate
        ev_log2 = -np.log(R) + L*np.log((1. - np.exp(-z))/(1. - np.exp(-z*r)))
        return np.exp(ev_log2)       
    
    
def function_fit(k, mu, r=5):
    # Function to fit experimental data (circles). See SI Eq (23)
    x= np.exp(mu)
    k = float(k)
    r = float(r)
    if abs(x - 1.) < 1e-6:
	# Taylor expansion of quotient up to first order at x=1 see Methods
	return k/r + (k*(x-1.)*(k-r))/(2.*r) 
    else:
	return (x**k - 1.)/(x**r - 1.)

def save_obj(obj, name ):
    # Saves python dictionary to file
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    # Loads python dictionary from file
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

def plot_circles(dic_value, save = False, filename = 'Roses_fitting', title = '' ):
    '''
    dic_value: Dictionary with the following structure: {case: [(fitted mu, lower limit of CI, upper limit of CI),list of layers],...}
    If save = True, the plots are saved in pdf with name "filename"
    title: optional title for the figure
    '''
    l = dic_value[1]
    mu = dic_value[0][0]
    t1 = dic_value[0][1]
    t2 = dic_value[0][2]
    r = len(l)

    xdata = range(1,r+1)
    ydata = [float(x) / np.cumsum(l)[-1] for x in np.cumsum(l)] # experimental
    domain = np.linspace(0.5,r,30)
    curve = [function_fit(t,mu, r) for t in domain]             # fitted     
    curve1 = [function_fit(t,t1, r) for t in domain]            # lower bound of CI
    curve2 = [function_fit(t,t2, r) for t in domain]            # upper bound of CI
    if title != '':	
    	plt.title(title)
    plt.ylabel('fraction of actors')
    circles = ['circle 1', 'circle 2', 'circle 3', 'circle 4', 'circle 5', 'circle 6']
    labels = tuple(circles[:r+1])
    plt.xticks(range(1,r+1),labels ,rotation=40  )
    plt.ylim((0,1.1))

    plt.plot(domain, curve1, linestyle ='--', color = 'w', alpha = 0.)
    plt.plot(domain, curve2, linestyle ='--', color = 'w', alpha = 0.)

    plt.plot(domain, curve, linestyle ='--')
    plt.scatter(xdata,ydata,s=2)
    plt.fill_between(domain, curve1, curve2, alpha = 0.25)
    if save == True:
        plt.savefig(filename + '.pdf', dpi=300)
    plt.show()
