
# Induced seismicity models
# -------------------------
# Python classes for modeling induced seismicity in 1-3 dimensions for point injection sources, before and after a rate reduction.
#
# Supplementary material to Dempsey and Riffault (2018) "Response of induced seismicity to injection rate reduction: models of delay, decay, quiescence, recovery and Oklahoma"
#
# Authors
# -------
# David Dempsey, Jeremy Riffault
#
# Instructions
# ------------
# User will need to install Python (2.X or 3.X) and supporting packages.

 
# Imports
import numpy as np
from scipy.optimize import root, brenth
from scipy.interpolate import interp2d
from scipy.special import expi

EMC = 0.57721566490153286060651209008240243104215933593992   # Euler-Mascheroni constant

class InducedSeismicity(object):
	''' Create an induced seismicity model object.
	
		Parameters
		----------
		f : float
			Fraction to cut injection rate.
		dim : int
			Flow dimension (1, 2 or 3).
		pcrit : float, optional
			Critical pressure for earthquake triggering (default = 0).
		sim : dict, optional
			Reservoir simulation output. Dictionary comprises vector of distances, r, 
			vector of times, t, and matrix of simulated fluid pressure increases, p. 
			Dimension of p must conform with r and t. If sim is not None, ignores dim.
		
	'''
	def __init__(self, f, dim, pcrit=0, sim = None):
		# error checking
		if dim == 1 and pcrit > 2/np.sqrt(np.pi):
			raise ValueError('No seismicity will occur before rate cut for 1D flow and p>1.128.')
		
		# assign variables
		self.f = f
		self.d = dim
		self.pc = pcrit
		
		if self.d != 2.5 and self.d != 1.5:
			# for point source, compute bifurcation point 
			self.tstar, self.rstar = self.apex()
			self.pstar = self.prt(self.tstar, self.rstar)
		else:
			# for reservoir simulator output
				# store output data
			self.r = sim['r']
			self.t = sim['t']
			self.p = sim['p']
				# set up pressure interpolator
			self.pint = interp2d(self.r, self.t, self.p.T, kind = 'linear', bounds_error=False, fill_value=None)
				# set up pressure gradient interpolators
			tt,rr = np.meshgrid(self.t,self.r)
					# in time
			dpdt = np.diff(self.p)/np.diff(tt)
			tmid = 0.5*(self.t[:-1]+self.t[1:])
			self.dpint = interp2d(self.r, tmid, dpdt.T, kind = 'linear', bounds_error=False, fill_value=None)
					# with distance
			dpdr = np.diff(self.p,axis=0)/np.diff(rr,axis=0)
			rmid = 0.5*(self.r[:-1]+self.r[1:])
			self.dpdrint = interp2d(rmid, self.t, dpdr.T, kind = 'linear', bounds_error=False, fill_value=None)
	def prt(self,t,r,p0=0): 					# p(r,t)
		''' Fluid pressure.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			p0 : float, optional
				Reference pressure for root solve.
				
			Returns
			-------
			p : array-like
				Pressure.
		
		'''
		# pass to dimension specific method
		if self.d == 1: 
			_prt = self._prt1
		elif self.d == 1.5: 
			_prt = self._prt1i
		elif self.d == 2: 
			_prt = self._prt2
		elif self.d == 2.5: 
			_prt = self._prt2i
		elif self.d == 3: 
			_prt = self._prt3
		return _prt(iterable(t),r,p0)
	def _prt1(self,t,r,p0):						#   - 1D point source
		''' Fluid pressure in 1 dimension.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			p0 : float, optional
				Reference pressure for root solve.
				
			Returns
			-------
			p : array-like
				Pressure.
				
			Notes
			-----
			Implements stepped 1D pressure solution.
			
			p = 2(sqrt(t/pi)*exp(-r^2/t)-r*erfc(r/sqrt(t))),	   			for t<=1
			p = 2(sqrt(t/pi)*exp(-r^2/t)-r*erfc(r/sqrt(t))) 
				- f*2(sqrt((t-1)/pi)*exp(-r^2/(t-1))-r*erfc(r/sqrt(t-1))),  for t>1
		'''
		# pressure due to first injection source
		pi = 2.*(np.sqrt(t/np.pi)*np.exp(-r**2/t)-r*erfc(r/np.sqrt(t)))

		# times at which modification required
		inds = np.where(t>1.)
		
		# pressure due to second injection source (rate reduction)
		pi[inds] += -2.*self.f*(np.sqrt((t[inds]-1.)/np.pi)*np.exp(-r**2/(t[inds]-1.))-r*erfc(x/np.sqrt(t[inds]-1.)))	

		# subtract reference pressure
		return pi - p0
	def _prt1i(self,t,r,p0):					#   - 1D arbitrary simulation output
		''' Fluid pressure in 1 dimension for arbitrary simulation output.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			p0 : float, optional
				Reference pressure for root solve.
				
			Returns
			-------
			p : array-like
				Pressure.
				
			Notes
			-----
			Interpolates solution from simulator output.
		'''
		return self.pint(r,t)
	def _prt2(self,t,r,p0):						#   - 2D point source (Theis)
		''' Fluid pressure in 2 dimensions.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			p0 : float, optional
				Reference pressure for root solve.
				
			Returns
			-------
			p : array-like
				Pressure.
				
			Notes
			-----
			Implements stepped 2D pressure solution (Theis solution).
			
			p = W(r^2/t),				   		for t<=1
			p = W(r^2/t) - f*W(r^2/(t-1)),  	for t>1
			
			where W(x) is the exponential integral.
		'''
		# pressure due to first injection source
		pi = -expi(-r**2/t)

		# times at which modification required
		inds = np.where(t>1.)
		
		# pressure due to second injection source (rate reduction)
		pi[inds] += -self.f*-expi(-r**2/(t[inds]-1.))	

		# subtract reference pressure
		return pi - p0
	def _prt2i(self,t,r,p0):					#   - 2D arbitrary simulation output
		''' Fluid pressure in 2 dimensions for arbitrary simulation output.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			p0 : float, optional
				Reference pressure for root solve.
				
			Returns
			-------
			p : array-like
				Pressure.
				
			Notes
			-----
			Interpolates solution from simulator output.
		'''
		return self.pint(r,t)
	def _prt3(self,t,r,p0):						#   - 3D point source
		''' Fluid pressure in 3 dimensions.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			p0 : float, optional
				Reference pressure for root solve.
				
			Returns
			-------
			p : array-like
				Pressure.
				
			Notes
			-----
			Implements stepped 3D pressure solution.
			
			p = erfc(r/sqrt(t))/r,	   						for t<=1
			p = erfc(r/sqrt(t))/r-f*erfc(r/sqrt(t-1))/r,  	for t>1
		'''
		# pressure due to first injection source
		pi = erfc(r/np.sqrt(t))/r
		
		# times at which modification required
		inds = np.where(t>1.)
		
		# pressure due to second injection source (rate reduction)
		pi[inds] += -self.f*erfc(r/np.sqrt(t[inds]-1.))/r
		
		# subtract reference pressure
		return pi - p0
	def dpdt(self,t,r,dpdt0=0):					# dp/dt(r,t)
		''' Fluid pressure partial derivative with time.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdt0 : float, optional
				Reference gradient for root solve.
				
			Returns
			-------
			dpdt : array-like
				Pressure partial derivative with time.
		
		'''
		# pass to dimension specific method
		if self.d == 1: 
			_dpdt = self._dpdt1
		elif self.d == 1.5: 
			_dpdt = self._dpdt1i
		elif self.d == 2: 
			_dpdt = self._dpdt2
		elif self.d == 2.5: 
			_dpdt = self._dpdt2i
		elif self.d == 3: 
			_dpdt = self._dpdt3
		return _dpdt(iterable(t),r,dpdt0)
	def _dpdt1(self,t,x,dpdt0):					#   - 1D point source
		''' Fluid pressure partial derivative with time in 1 dimension.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdt0 : float, optional
				Reference pressure gradient for root solve.
				
			Returns
			-------
			dpdt : array-like
				Pressure partial derivative with time.
				
			Notes
			-----
			Derivative of stepped 1D pressure solution.
			
			dpdt = exp(-x^2/t)/sqrt(pi*t),				 for t<=1
			dpdt = exp(-x^2/t)/sqrt(pi*t) 
				   - f*exp(-x^2/(t-1))/sqrt(pi*(t-1)),   for t>1
		'''
		# dpdt due to first injection source
		dpdti = np.exp(-x**2/t)/np.sqrt(np.pi*t)
		
		# times at which modification required
		inds = np.where(t>1.)
		
		# dpdt due to second injection source (rate reduction)
		dpdti[inds] += -self.f*np.exp(-x**2/(t[inds]-1.))/np.sqrt(np.pi*(t[inds]-1.))
		
		# subtract reference gradient
		return dpdti - dpdt0
	def _dpdt1i(self,t,r,dpdt0):				#   - 1D arbitrary simulation output
		''' Fluid pressure partial derivative with time in 1 dimensions for arbitrary simulation output.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdt0 : float, optional
				Reference pressure gradient for root solve.
				
			Returns
			-------
			dpdt : array-like
				Pressure partial derivative with time.
				
			Notes
			-----
			Interpolates solution from simulator output.
		'''
		return self.dpint(r,t)
	def _dpdt2(self,t,r,dpdt0):					#   - 2D point source (Theis)
		''' Fluid pressure partial derivative with time in 2 dimensions.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdt0 : float, optional
				Reference pressure gradient for root solve.
				
			Returns
			-------
			dpdt : array-like
				Pressure partial derivative with time.
				
			Notes
			-----
			Derivative of stepped 2D pressure solution (Theis solution).
			
			dpdt = exp(-r^2/t)/t,						   		for t<=1
			dpdt = exp(-r^2/t)/t - f*exp(-r^2/(t-1)) / (t-1),   for t>1
		'''
		# dpdt due to first injection source
		dpdti = np.exp(-r**2/t)/t
		
		# times at which modification required
		inds = np.where(t>1.)
		
		# dpdt due to second injection source (rate reduction)
		dpdti[inds] += -self.f*np.exp(-r**2/(t[inds]-1.))/(t[inds]-1.)
		
		# subtract reference gradient
		return dpdti - dpdt0
	def _dpdt2i(self,t,r,dpdt0):				#   - 2D arbitrary simulation output
		''' Fluid pressure partial derivative with time in 2 dimensions for arbitrary simulation output.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdt0 : float, optional
				Reference pressure gradient for root solve.
				
			Returns
			-------
			dpdt : array-like
				Pressure partial derivative with time.
				
			Notes
			-----
			Interpolates solution from simulator output.
		'''
		return self.dpint(r,t)
	def _dpdt3(self,t,r,dpdt0):					#   - 3D point source
		''' Fluid pressure partial derivative with time in 3 dimensions.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdt0 : float, optional
				Reference pressure gradient for root solve.
				
			Returns
			-------
			dpdt : array-like
				Pressure partial derivative with time.
				
			Notes
			-----
			Derivative of stepped 3D pressure solution.
			
			dpdt = exp(-r^2/t)/sqrt(pi)/t^1.5,				 	for t<=1
			dpdt = exp(-r^2/t)/sqrt(pi)/t^1.5 
					- f*exp(-r^2/(t-1))/sqrt(pi)/(t-1)^1.5,   	for t>1
		'''
		# dpdt due to first injection source
		dpdti = np.exp(-r**2/t)/np.sqrt(np.pi)/t**1.5
		
		# times at which modification required
		inds = np.where(t>1.)
		
		# dpdt due to second injection source (rate reduction)
		dpdti[inds] += -self.f*np.exp(-r**2/(t[inds]-1.))/np.sqrt(np.pi)/(t[inds]-1.)**1.5
		
		# subtract reference gradient
		return dpdti - dpdt0
	def dpdtn(self,t,r,dpdt0):					# -dp/dt(r,t)
		''' Negative of fluid pressure partial derivative with time.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdt0 : float, optional
				Reference gradient for root solve.
				
			Returns
			-------
			dpdt : array-like
				Pressure partial derivative with time.
		
			Notes
			-----
			Required for root solve.
		'''
		return -self.dpdt(t,r,dpdt0)
	def ptr(self,r,t,p0=0):						# p(t,r)
		''' Fluid pressure.
			
			Parameters
			----------
			r : float
				Distance from injection source.
			t : array-like
				Time.
			p0 : float, optional
				Reference pressure for root solve.
				
			Returns
			-------
			p : array-like
				Pressure.
		
			Notes
			-----
			Same as method prt() except order of arguments reversed.
		
		'''
		return self.prt(t,r,p0)
	def dpdr(self,r,t,dpdr0=0):					# dp/dr(r,t)
		''' Fluid pressure partial derivative with distance.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdr0 : float, optional
				Reference gradient for root solve.
				
			Returns
			-------
			dpdr : array-like
				Pressure partial derivative with distance.
		
		'''
		# pass to dimension specific method
		if self.d == 1:
			_dpdr=self._dpdr1
		elif self.d == 1.5: 
			_dpdr = self._dpdr1i
		elif self.d == 2:
			_dpdr=self._dpdr2
		elif self.d == 2.5: 
			_dpdr = self._dpdr2i
		elif self.d == 3:
			_dpdr=self._dpdr3
		return _dpdr(r,iterable(t),dpdr0)
	def _dpdr1(self,x,t,dpdr0=0):				#	- 1D point source 
		''' Fluid pressure partial derivative with distance in 1 dimension.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdr0 : float, optional
				Reference pressure gradient for root solve.
				
			Returns
			-------
			dpdr : array-like
				Pressure partial derivative with distance.
				
			Notes
			-----
			Derivative of stepped 1D pressure solution.
			
			dpdr = -2*erfc(-x/sqrt(t)),						 	 for t<=1
			dpdr = -2*erfc(-x/sqrt(t)) + 2*f*erfc(-x/sqrt(t-1)), for t>1
				
		'''
		# dpdr due to first injection source
		dpdri = -2.*erfc(x/np.sqrt(t))
		
		# times at which modification required
		inds = np.where(t>1.)
		
		# dpdr due to second injection source (rate reduction)
		dpdri[inds] += 2.*self.f*erfc(x/np.sqrt(t-1.))
		
		# subtract reference gradient
		return dpdri - dpdr0
	def _dpdr1i(self,t,r,dpdr0=0):				#	- 1D arbitrary simulation output
		''' Fluid pressure partial derivative with distance in 2 dimensions for arbitrary simulation output.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdr0 : float, optional
				Reference pressure gradient for root solve.
				
			Returns
			-------
			dpdr : array-like
				Pressure partial derivative with distance.
				
			Notes
			-----
			Interpolates solution from simulator output.
		'''
		return self.dpdrint(r,t)
	def _dpdr2(self,r,t,dpdr0=0):				# 	- 2D point source (Theis)
		''' Fluid pressure partial derivative with distance in 2 dimensions.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdr0 : float, optional
				Reference pressure gradient for root solve.
				
			Returns
			-------
			dpdr : array-like
				Pressure partial derivative with distance.
				
			Notes
			-----
			Derivative of stepped 2D pressure solution (Theis solution).
			
			dpdr = -2*exp(-r^2/t),						 	for t<=1
			dpdr = -2*(exp(-r^2/t) + 2*f*exp(-r^2/(t-1))),  for t>1
		'''
		# dpdr due to first injection source
		dpdri = -2.*np.exp(-r**2/t)
		
		# times at which modification required
		inds = np.where(t>1.)
		
		# dpdr due to second injection source (rate reduction)
		dpdri[inds] += 2.*self.f*np.exp(-r**2/(t[inds]-1.))
		
		# subtract reference gradient
		return dpdri - dpdr0
	def _dpdr2i(self,t,r,dpdr0=0):				#	- 2D arbitrary simulation output
		''' Fluid pressure partial derivative with distance in 2 dimensions for distributed injection source.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdr0 : float, optional
				Reference pressure gradient for root solve.
				
			Returns
			-------
			dpdr : array-like
				Pressure partial derivative with distance.
				
			Notes
			-----
			Interpolates solution from reservoir simulator output.
		'''
		return self.dpdrint(r,t)
	def _dpdr3(self,r,t,dpdr0=0):				# 	- 3D point source
		''' Fluid pressure partial derivative with distance in 3 dimensions.
			
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance from injection source.
			dpdr0 : float, optional
				Reference pressure gradient for root solve.
				
			Returns
			-------
			dpdr : array-like
				Pressure partial derivative with distance.
				
			Notes
			-----
			Derivative of stepped 3D pressure solution.
			
			dpdt = exp(-r^2/t)/sqrt(pi)/t^1.5,				 	for t<=1
			dpdt = exp(-r^2/t)/sqrt(pi)/t^1.5 
					- f*exp(-r^2/(t-1))/sqrt(pi)/(t-1)^1.5,   	for t>1
		'''
		# dpdr due to first injection source
		dpdri = -erfc(r/np.sqrt(t))/r**2 - 2.*np.exp(-r**2/t)/np.sqrt(np.pi*t)/r
		
		# times at which modification required
		inds = np.where(t>1.)
		
		# dpdr due to injection source (rate reduction)
		dpdri[inds] += -self.f*(-erfc(r/np.sqrt(t[inds]-1.))/r**2 - 2.*np.exp(-r**2/(t[inds]-1.))/np.sqrt(np.pi*(t[inds]-1.))/r)
		
		# subtract reference gradient
		return dpdri - dpdr0
	def rp(self,t,p):							# r(p,t)
		''' Distance a given fluid pressure is found at given time.
		
			Parameters
			----------
			t : array-like
				Time.
			p : float
				Pressure.
				
			Returns
			-------
			r : array-like
				Distance.
				
			Notes
			-----
			Pressure solutions not generally invertible, therefore distance determined by root solve.
		'''
		# convert time to iterable if not already
		t = iterable(t)
		
		# error checks
		if p <= 0:
			return 1.e32+0.*t
			
		r = []
		for ti in t:
			# check for special case of max pressure
			if self.d == 1:
				pmax = self.prt(ti, 0.)
				if p>pmax:
					r.append(np.float('NaN'))
					continue
			r.append(root(self.ptr, 1.e-5, (ti,p), jac=self.dpdr).x)
				
		r = np.array(r).flatten()
		return r
	def gt(self,t,r0=0):						# r=g(t) for dpdt=0
		''' Distance where dpdt=0 for given time, function g(t).
		
			Parameters
			----------
			t : array-like
				Time.
			r0 : float, optional
				Reference distance (for root solve)
				
			Returns
			-------
			r : array-like
				Distance.
				
			Notes
			-----
			Multidimensional backfront equation modified for rate cut,
			
			r = sqrt(t*(t-1)*ln(f^(2/d)*t/(t-1))d/2),
			
			where d is dimension.
		'''
		# convert time to iterable if not already
		t = iterable(t)
		
		# error check
		if any(t<1.):
			raise ValueError
		if any(t>1./(1.-self.f**(2./self.d))):
			raise ValueError
			
		return np.sqrt(np.log(self.f**(2./self.d)*t/(t-1.))*(t*(t-1.))*self.d/2.) - r0
	def dgdt(self,t,dgdt0=0):					# dg/dt(t)
		''' Partial derivative with time of g(t).
		
			Parameters
			----------
			t : float
				Time
				
			Returns
			-------
			dgdt : float
				Partial derivative with time.
		'''
		lnfdt = np.log(t*self.f**(2./self.d)/(t-1.))
		return self.d*((2.*t-1.)*lnfdt-1.)/np.sqrt(8.*self.d*t*(t-1.)*lnfdt)
	def gr(self,r):								# t=g^-1(r)
		''' Time where dpdt=0 for given distance, function g^-1(r).
		
			Parameters
			----------
			r : float
				Distance.
				
			Returns
			-------
			t : list
				Two item list containing time of maximum and minimum pressure.
				
			Notes
			-----
			Uses root solve to invert two solutions of multidimensional backfront equation.
			
			Returns NaN if r above path maximum.
		'''
		# if r above limit of path, return NaN
		if r > self.rstar:
			return float('NaN'), float('NaN')
		
		# if r << 1, return 1 and trmax
		if r < 1.e-5:
			return 1, 1/(1-self.f**(2./self.d))
		
		# compute two roots
		root1 = root(self.gt, 1.+1.e-8, (r,), jac = self.dgdt).x
		root2 = root(self.gt, 1./(1.-self.f**(2./self.d))-1.e-8, (r,), jac = self.dgdt).x
		return root1, root2
	def gp0(self,ta,p):
		''' Root equation for method gp.
			
			Parameters
			----------
			ta : float
				Time at possible point A.
			p : float
				Pressure.
		'''
		ra = self.gt(ta)
		return self.prt(ta,ra)-p
	def gp(self,p):								# solve p(g(ta),ta) = pcrit
		''' Time and distance where dpdt=0 for fluid pressure. 
		
			Parameters
			----------
			p : float
				Pressure.
				
			Returns
			-------
			t : float
				Time.
			r : float
				Distance.
			
			Notes
			-----
			Returns first root (point A) or NaN if p less than bifurcation point pressure.
			Computed using root solve.
		'''
		# if p less than bifurcation pressure, return NaN
		if p < self.pstar:
			return float('NaN')
		
		# compute root
		ta = brenth(self.gp0, 1.+1.e-8, self.tstar, args=(p,))
		return ta, self.gt(ta)
	def h0(self,th,t): 
		''' Root equation for method ht.
			
			Parameters
			----------
			th : float
				Time on curve h(t).
			t : float
				Time of curve g(t).
		'''
		r = self.gt(th)
		return self.prt(th,r)-self.prt(t,r)
	def ht(self,t):								# r=h(t) for p=pmax(r(t))
		''' Distance on curve h(t).
		
			Parameters
			----------
			t : float
				Time.
				
			Returns
			-------
			r : float
				Distance.
		
			Notes
			-----
			Approximation is used for cases where t is too close to boundaries
		'''
		# convert time to iterable if not already
		t = iterable(t)
			
		# error check
		if any(t<self.tstar):
			raise ValueError

		# compute h		 
		out = []
		for ti in t:
			# check if approximation should be used
			if (self.h0(1.+1.e-8,ti)>0 and self.h0(self.tstar-1.e-8,ti)<0):
			
				# root solve
				th = np.array([brenth(self.h0, 1.+1.e-8, self.tstar-1.e-8, args=(ti,)),])
				out.append(self.gt(th)[0])
			else:
				
				# approximation valid for 
				if self.d == 1:
					out.append(1.e-32) 
				elif self.d == 2:
					out.append(np.sqrt((ti-1.)*ti**-(1/self.f)*(np.exp(-EMC)))) 
				elif self.d == 3:
					out.append(1.e-32) 
		return out
	def hr(self,r):								# t=h^-1(r) for p(t,r)=pmax(r)
		''' Time on curve h(t).
		
			Parameters
			----------
			r : float
				Distance.
				
			Returns
			-------
			t : float
				Time.
				
			Notes
			-----
			Uses root solve.
		'''

		# error check
		if r>self.rstar:
			raise ValueError
			
		tmax,tmin = self.gr(r)
		
		return root(self.prt, tmin+1.e-5, (r,self.prt(tmax,r)), jac=self.dpdtn, method = 'hybr').x
	def hp(self,p):								# h^-1(r) where p(t,r)=pmax(r)
		''' Compute point on curve r=h(t) at given pressure.
		
			Parameters
			----------
			p : float
				Pressure.
				
			Returns
			-------
			t : float
				Time.
			r : float
				Distance.
		'''  

		# if p below limit of path, return NaN
		if p <self.pstar:
			return float('NaN')
		
		# get point a
		ta,ra = self.gp(p)
		tb = self.hr(ra)
		return tb, ra
	def apex(self):								# locate bifurcation point
		''' Compute bifurcation point.
		'''
		# find maximum of path dgdt=0
		tstar = brenth(self.dgdt, 1.+1.e-5, 1./(1.-self.f**(2./self.d))-1.e-5)
		return [tstar, self.gt(np.array([tstar,]))]
	def nrt(self,t,r):							# n(r,t)
		''' Compute local seismicity rate.
		
			Parameters
			----------
			t : array-like
				Time.
			r : float
				Distance.
				
			Returns
			-------
			n : array-like
				Seismicity rate.
		
			Notes
			-----
			Seismicity rate normalised to 1 at r=1 and t=1.
		'''
		# convert time to iterable if not already
		t = iterable(t)
		
		# normalising factor
		n0 = self.dpdt(1,1)
		
		# case 1: before rate cut
		if t<=1:
			# check if critical pressure given
			if self.pc>0.:
				rc = self.rp(t,self.pc)
				# check if outside critical distance, in which case zero seismicity
				if r>rc:
					return 1.e-32
					
			# inside critical distance, compute seismicity rate
			return self.dpdt(t,r)/n0
			
		# case 2: after rate cut
		else:
			# determine which part of the r-t diagram we are in
			
			# case 2a: cessation possible
			if self.pstar<self.pc:
				# compute cessation interval
				ta, ra = self.gp(self.pc)
				tb, rb = self.hp(self.pc)
				
				# check if in cessation interval, in which case zero seismicity
				if ta<=t<=tb:
					return 1.e-32
			else:
				ta = self.tstar
				tb = self.tstar
					
			# case 2b: seismicity declining after rate-cut
			if 1.<t<ta: 
				# inside envelope, compute seismicity rate
				if self.gt(t) < r < self.rp(t,self.pc):
					return self.dpdt(t,r)/n0
					
				# outside envelope, in which case zero seismicity
				else:
					return 1.e-32
					
			# case 2c: seismicity return after decline 
			elif t>tb:
				# inside envelope, compute seismicity rate
				if self.ht(t) < r < self.rp(t,self.pc):
					return self.dpdt(t,r)/n0
					
				# outside envelope, in which case zero seismicity
				else:
					return 1.e-32
	def nt(self,t,rmax=None):					# n(t)
		''' Compute global seismicity rate.
		
			Parameters
			----------
			t : array-like
				Time.
			rmax : float, optional
				Maximum distance to integrate seismicity.
			
			Returns
			-------
			n : array-like
				Seismicity rate.
		
			Notes
			-----
			Seismicity rate normalised to 1 at r=1 and t=1.
		'''
		# pass to dimension specific method
		if self.d == 1:
			_nt=self._nt1
		elif self.d == 1.5:
			_nt=self._nt1i
		elif self.d == 2:
			_nt=self._nt2
		elif self.d == 2.5:
			_nt=self._nt2i
		elif self.d == 3:
			_nt=self._nt3
		return _nt(iterable(t),rmax)
	def _nt1(self,t,rmax):						#   - 1D point source
		''' Compute global seismicity rate.
		
			Parameters
			----------
			t : array-like
				Time.
			rmax : float, optional
				Maximum distance to integrate seismicity.
			
			Returns
			-------
			n : array-like
				Seismicity rate.
		
			Notes
			-----
			Seismicity rate normalised to 1 at r=1 and t=1.
		'''
		# initialization
		n = 0.*t	  
		inds1 = np.where(t<=1.)
		inds2 = np.where(t>1.)

		# lower bound r0
		indsg = np.where((t>1.) & (t<=self.tstar))
		indsh = np.where(self.tstar<t)
		r0 = np.concatenate([0.*inds1[0],self.gt(t[indsg]),self.ht(t[indsh])])
		
		# case with rcrit=r(t)
		if self.pc>0. or rmax is not None:
			if self.pc>0.:
				r1 = self.rp(t, self.pc)
			else:
				r1 = 1.e32+0.*t
			if rmax is not None:
				r1[np.where(r1>rmax)] = rmax
		else:
			r1 = 1.e32+0.*t
				
		# case with rcrit=infinity
		n[inds1] = 0.5*(erf(r1[inds1]/np.sqrt(t[inds1]))-erf(r0[inds1]/np.sqrt(t[inds1])))
		n[inds2] = 0.5*(erf(r1[inds2]/np.sqrt(t[inds2]))-erf(r0[inds2]/np.sqrt(t[inds2]))-self.f*erf(r1[inds2]/np.sqrt((t[inds2]-1)))+self.f*erf(r0[inds2]/np.sqrt((t[inds2]-1))))

		# compute seismicity at rate cut and normalise
		if self.pc>0. or rmax is not None:
			if self.pc>0.:
				rcut = self.rp(1., self.pc)
			else:
				rcut = 1.e32+0.*t
			if rmax is not None:
				if rcut > rmax: 
					rcut = rmax
		else:
			rcut = 1.e32

		n0 = 0.5*(erf(rcut)-erf(0.))
		if abs(n0) < 1.e-10:
			raise ValueError('seismicity at rate cut is zero')
		
		n = n/n0
		
		# set seismicity to 0 where rcrit<r0
		inds3 = np.where(r1<=r0)
		n[inds3] = 0.
		
		inds3 = np.where(r1!=r1)
		n[inds3] = 0.
		
		return np.array(n)
	def _nt1i(self,t,rmax):						#   - 1D arbitrary simulation output
		''' Compute global seismicity rate.
		
			Parameters
			----------
			t : array-like
				Time.
			rmax : float, optional
				Maximum distance to integrate seismicity.
			
			Returns
			-------
			n : array-like
				Seismicity rate.
		'''
		# set up solution grid
		rs = np.linspace(self.r[0], self.r[-1], 201)
		rm = 0.5*(rs[:-1] + rs[1:])
		dr = -0.5*(rs[:-1] - rs[1:])
		tt,rr = np.meshgrid(t,rm)
		dr = np.tile(dr,reps=(len(t),1)).T
		if rmax is not None:
			if rmax>0.:
				rr[rr>rmax]=0.
			elif rmax<0.:
				rr[rr<abs(rmax)]=0.
		
		# compute pressure and gradients
		p = 0.*tt
		dp = 0.*tt
		dpdr = 0.*tt
		for i in range(tt.shape[0]):
			p[i,:] = self.prt(t, rm[i]).T[0]
			dp[i,:] = self.dpdt(t, rm[i]).T[0]
			dpdr[i,:] = self.dpdr(t, rm[i]).T[0]
		
		mask = 1.+0.*dp
		
		# interpolation scheme
		for i in range(mask.shape[0]):
			pmax1 = self.pc*1.
			pmax2 = self.pc*1.
			for j in range(mask.shape[1]):
				dpdri = dpdr[i,j]
				pedge1 = p[i,j]-dpdri*dr[i,j]/2.
				pedge2 = p[i,j]+dpdri*dr[i,j]/2.
					
				if pedge2<pmax2 and pedge1<pmax1:
					mask[i,j] = 0.
				elif pedge2>pmax2 and pedge1<pmax1:
					mask[i,j] = 1.-(pmax1-pedge1)/(pedge2-pedge1-(pmax2-pmax1))
					pmax2 = pedge2
				elif pedge2<pmax2 and pedge1>pmax1:
					mask[i,j] = (pmax1-pedge1)/(pedge2-pedge1-(pmax2-pmax1))
					pmax1 = pedge1
				else:
					pmax2 = pedge2
					pmax1 = pedge1
					
		n = np.sum(dp*mask*dr, axis=0)
			
		return n
	def _nt2(self,t,rmax):						#   - 2D point source (Theis)
		''' Compute global seismicity rate.
		
			Parameters
			----------
			t : array-like
				Time.
			rmax : float, optional
				Maximum distance to integrate seismicity.
			
			Returns
			-------
			n : array-like
				Seismicity rate.
		
			Notes
			-----
			Seismicity rate normalised to 1 at r=1 and t=1.
		'''
		# initialization
		n = 0.*t	  
		inds1 = np.where(t<=1.)
		inds2 = np.where(t>1.)

		# lower bound r0
		indsg = np.where((t>1.) & (t<=self.tstar))
		indsh = np.where(self.tstar<t)
		r0 = np.concatenate([0.*inds1[0],self.gt(t[indsg]),self.ht(t[indsh])])

		# case with rcrit=infinity
		n[inds1] += 1.
		n[inds2] += np.exp(-r0[inds2]**2/t[inds2])-self.f*np.exp(-r0[inds2]**2/(t[inds2]-1.))

		# case with rcrit=r(t)
		if self.pc>0. or rmax is not None:
			if self.pc>0.:
				rc = self.rp(t,self.pc)
			else:
				rc = 1.e32+0.*t
			if rmax is not None:
				rc[np.where(rc>rmax)] = rmax
			n[inds1] += -np.exp(-rc[inds1]**2/t[inds1])
			n[inds2] += -np.exp(-rc[inds2]**2/t[inds2])+self.f*np.exp(-rc[inds2]**2/(t[inds2]-1.))

			# set seismicity to 0 where rcrit<r0
			inds3 = np.where(rc<=r0)
			n[inds3] = 0.
			
		# compute seismicity at rate cut and normalise
		n0 = 1
		if self.pc>0. or rmax is not None:
			if self.pc>0.:
				rcut = self.rp(1., self.pc)
			else:
				rcut = 1.e32+0.*t
			if rmax is not None:
				if rcut > rmax: 
					rcut = rmax
			n0 += -np.exp(-rcut**2)
		if abs(n0) < 1.e-10:
			raise ValueError('seismicity at rate cut is zero')
		
		n = n/n0
			
		return np.array(n)
	def _nt2i(self,t,rmax):						#   - 2D arbitrary simulation output
		''' Compute global seismicity rate.
		
			Parameters
			----------
			t : array-like
				Time.
			rmax : float, optional
				Maximum distance to integrate seismicity.
			
			Returns
			-------
			n : array-like
				Seismicity rate.
		
			Notes
			-----
			Seismicity rate normalised to 1 at r=1 and t=1.
		'''
		# set up radial solution grid
		rs = np.logspace(-2, np.log10(self.r[-1]), 201)
		rm = 0.5*(rs[:-1] + rs[1:])
		dr = -0.5*(rs[:-1] - rs[1:])
		tt,rr = np.meshgrid(t,rm)
		dr = np.tile(dr,reps=(len(t),1)).T
		if rmax is not None:
			if rmax>0.:
				rr[rr>rmax]=0.
			elif rmax<0.:
				rr[rr<abs(rmax)]=0.
		
		# compute pressure and gradients
		p = 0.*tt
		dp = 0.*tt
		dpdr = 0.*tt
		for i in range(tt.shape[0]):
			p[i,:] = self.prt(t, rm[i]).T[0]
			dp[i,:] = self.dpdt(t, rm[i]).T[0]
			dpdr[i,:] = self.dpdr(t, rm[i]).T[0]
		
		mask = 1.+0.*dp
		
		# interpolation scheme
		for i in range(mask.shape[0]):
			pmax1 = self.pc*1.
			pmax2 = self.pc*1.
			for j in range(mask.shape[1]):
				dpdri = dpdr[i,j]
				pedge1 = p[i,j]-dpdri*dr[i,j]/2.
				pedge2 = p[i,j]+dpdri*dr[i,j]/2.
					
				if pedge2<pmax2 and pedge1<pmax1:
					mask[i,j] = 0.
				elif pedge2>pmax2 and pedge1<pmax1:
					mask[i,j] = 1.-(pmax1-pedge1)/(pedge2-pedge1-(pmax2-pmax1))
					pmax2 = pedge2
				elif pedge2<pmax2 and pedge1>pmax1:
					mask[i,j] = (pmax1-pedge1)/(pedge2-pedge1-(pmax2-pmax1))
					pmax1 = pedge1
				else:
					pmax2 = pedge2
					pmax1 = pedge1
					
		n = 2*np.pi*np.sum(rr*dp*mask*dr, axis=0)
			
		return n
	def _nt3(self,t,rmax):						#   - 3D point source
		''' Compute global seismicity rate.
		
			Parameters
			----------
			t : array-like
				Time.
			rmax : float, optional
				Maximum distance to integrate seismicity.
			
			Returns
			-------
			n : array-like
				Seismicity rate.
		
			Notes
			-----
			Seismicity rate normalised to 1 at r=1 and t=1.
		'''
		# initialization
		n = 0.*t	  
		inds1 = np.where(t<=1)
		inds2 = np.where(t>1)

		# lower bound r0
		indsg = np.where((t>1) & (t<=self.tstar))
		indsh = np.where(self.tstar<t)
		r0 = np.concatenate([0.*inds1[0],self.gt(t[indsg]),self.ht(t[indsh])])
		
		# case with rcrit=r(t)
		if self.pc>0. or rmax is not None:
			if self.pc>0.:
				r1 = self.rp(t, self.pc)
			else:
				r1 = 1.e32+0.*t
			if rmax is not None:
				r1[np.where(r1>rmax)] = rmax
		else:
			r1 = 1.e32+0.*t
				
		# case with rcrit=infinity
		n[inds1] = 0.25*(erf(r1[inds1]/np.sqrt(t[inds1]))-2*r1[inds1]/np.sqrt(np.pi*t[inds1])*np.exp(-r1[inds1]**2/t[inds1])) - 0.25*(erf(r0[inds1]/np.sqrt(t[inds1]))-2*r0[inds1]/np.sqrt(np.pi*t[inds1])*np.exp(-r0[inds1]**2/t[inds1]))
		n[inds2] = 0.25*(erf(r1[inds2]/np.sqrt(t[inds2]))-2*r1[inds2]/np.sqrt(np.pi*t[inds2])*np.exp(-r1[inds2]**2/t[inds2])) - 0.25*(erf(r0[inds2]/np.sqrt(t[inds2]))-2*r0[inds2]/np.sqrt(np.pi*t[inds2])*np.exp(-r0[inds2]**2/t[inds2])) - 0.25*self.f*(erf(r1[inds2]/np.sqrt(t[inds2]-1))-2*r1[inds2]/np.sqrt(np.pi*(t[inds2]-1))*np.exp(-r1[inds2]**2/(t[inds2]-1))) + 0.25*self.f*(erf(r0[inds2]/np.sqrt(t[inds2]-1))-2*r0[inds2]/np.sqrt(np.pi*(t[inds2]-1))*np.exp(-r0[inds2]**2/(t[inds2]-1)))

		# compute seismicity at rate cut and normalise
		if self.pc>0. or rmax is not None:
			if self.pc>0.:
				rcut = self.rp(1., self.pc)
			else:
				rcut = 1.e32+0.*t
			if rmax is not None:
				if rcut > rmax: 
					rcut = rmax
		else:
			rcut = 1.e32

		n0 = 0.25*(erf(rcut)-2*rcut/np.sqrt(np.pi)*np.exp(-rcut**2))
		if abs(n0) < 1.e-10:
			raise ValueError('seismicity at rate cut is zero')
		
		n = n/n0
		
		# set seismicity to 0 where rcrit<r0
		inds3 = np.where(r1<=r0)
		n[inds3] = 0.
		
		inds3 = np.where(r1!=r1)
		n[inds3] = 0.
		
		return np.array(n)
	
def iterable(x):
	# convert vector to iterable if not already
	try:
		_ = (xi for xi in x)
		return x
	except TypeError:
		return np.array([x,])