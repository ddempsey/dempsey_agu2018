
import numpy as np
from inducedseismicity import *
from copy import copy
from matplotlib import pyplot as plt
import emcee, corner

class Fault(object):
	''' Fluid pressure simulation on 2D plane with regular grid.
	
		Pressure simulation data should be given in 5-column file containing
		v [m], w[m], t[s], dP[MPa]
		
		where v is along-fault coordinate and w is down-dip coordinate.		
	'''
	def __init__(self, filename):
		self.filename = filename
		self._load()
	def _load(self):
		''' loads in pressure data from file
		'''
		v,w,t,p = np.genfromtxt(self.filename).T
		self._grid(v,w,t,p)
	def _grid(self,v,w,t,p):
		''' constructs the grid of space and time coordinates and reshapes pressure data
		'''
		self.vu = np.unique(v)
		self.wu = np.unique(w)
		self.tu = np.unique(t)
		assert len(self.vu)*len(self.wu)*len(self.tu) == len(v)
		# lengths
		self.iv = self.vu.shape[0]
		self.iw = self.wu.shape[0]
		self.it = self.tu.shape[0]
		# spans
		self.dv = np.max(self.vu)-np.min(self.vu)
		self.dw = np.max(self.wu)-np.min(self.wu)
		self.dt = np.max(self.tu)-np.min(self.tu)

		# determine reshape operation appropriate to data
		foundReshape = False
		for i in range(6):
			# reshape dimensions to try
			if i == 0: 
				shp = [self.iv,self.iw,self.it]
				pmt = (0,1,2)
			elif i == 1: 
				shp = [self.iv,self.it,self.iw]
				pmt = (0,2,1)
			elif i == 2: 
				shp = [self.iw,self.it,self.iv]
				pmt = (2,0,1)
			elif i == 3: 
				shp = [self.iw,self.iv,self.it]
				pmt = (1,0,2)
			elif i == 4: 
				shp = [self.it,self.iw,self.iv]
				pmt = (2,1,0)
			elif i == 5: 
				shp = [self.it,self.iv,self.iw]
				pmt = (1,2,0)
			
			# reshape and check axis data differences
			dv = dim_diffs(v.flatten().reshape(shp))
			dw = dim_diffs(w.flatten().reshape(shp))
			dt = dim_diffs(t.flatten().reshape(shp))

			# for all data, check if differences along one dimension only
			if all(np.array([np.sum(dx < 1.e-6) for dx in [dv,dw,dt]]) == 2):
				foundReshape = True
				break
		assert foundReshape
		
		# reshape and permute data structures so variability in v,w,t along 0,1,2 axes
		self.v = v.flatten().reshape(shp).transpose(pmt)
		self.w = w.flatten().reshape(shp).transpose(pmt)
		self.t = t.flatten().reshape(shp).transpose(pmt)
		self.p = p.flatten().reshape(shp).transpose(pmt)
	def plot(self, save, v = None, w = None, t=None, clim = None):
		''' simple slice plotting **incomplete**
		'''
		# check plot dimension
		vars = ['v','w','t']
		if v is not None:
			dim = 0
		elif w is not None:
			dim = 1
		elif t is not None:
			dim = 2
		else:
			raise TypeError('need to specify plot slice dimension v, w, or t')
		x = (v,w,t)[dim]
		assert (x>np.min(self.__getattribute__(vars[dim]+'u')) and v<np.max(self.__getattribute__(vars[dim]+'u')))
		
		# **do later** linear interpolation
		# for now, find nearest value to specified v,w,t - plot slice

		f,ax = plt.subplots(1,1,figsize=(8,8))
		ix = np.argmin(abs(x - self.__getattribute__(vars[dim]+'u')))
		if dim == 0:
			dp = self.p[ix,:,:]
			x,y = self.tu,self.wu
			xl,yl = 'time','along-dip'
		elif dim == 1:
			dp = self.p[:,ix,:]
			x,y = self.tu,self.vu
			xl,yl = 'time','along-strike'
		else:
			dp = self.p[:,:,ix].T
			x,y = self.vu,self.wu
			xl,yl = 'along-strike','along-dip'
		
		levels = np.linspace(np.min(dp), np.max(dp), 11)
		if clim is not None:
			levels = np.linspace(clim[0], clim[1], 11)
		cax = ax.contourf(x,y,dp,levels=levels)
		ax.set_xlabel(xl)
		ax.set_ylabel(yl)
		plt.colorbar(cax)

		plt.savefig(save, dpi=400)

def dim_diffs(x):
	dx2 = np.max(x[0,0,:])-np.min(x[0,0,:])
	dx1 = np.max(x[0,:,0])-np.min(x[0,:,0])
	dx0 = np.max(x[:,0,0])-np.min(x[:,0,0])
	return np.array([dx0,dx1,dx2])

def simulate_seismicity(ti, dat, pc = 0, k = 1, wgt = None):
	''' compute bulk induced seismicity for fault **incomplete**

		parameters:
		----------
		ti : ndarray
			times at which to compute seismicity rate
		dat : Fault
			object containing simulation data, grid structure
		pc : float, optional
			critical fluid pressure, default = 0
		k : float, optional
			seismicity rate scaling factor, default = 1
		wgt : callable, optional
			weighting function for depth, default is constant
		
		returns:
		-------
		n : ndarray
			seismicity rate
	'''

	if wgt is None:
		wgt = lambda x: 1.

	# compute depth weights
	ps = [dat.p[:,0,:]]
	ws = [gauss_int(wgt, dat.wu[0], 0.5*np.sum(dat.wu[:2]))]
	for i,wi in enumerate(dat.wu[1:-1]):
		ps.append(dat.p[:,i+1,:])
		w1 = np.mean(dat.wu[i:i+2])
		w2 = np.mean(dat.wu[i+1:i+3])
		ws.append(gauss_int(wgt, w1, w2))

	n = 0.*ti
	for p,w in zip(ps,ws):
		iso = InducedSeismicity(f=0.99, dim=1.5, sim = {'r':dat.vu-dat.vu[0],'t':dat.tu,'p':p}, pcrit = pc)
		ni = iso.nt(ti)*w
		n += ni

	# **to do** compute an appropriate background rate for 
	nbackground = 1.e-20
	return n*k + nbackground
		
def gauss_int(f,x1,x2):
	''' estimate integratal of function F between X1 and X2 using Gaussian quadrature

		**have not verified or tested for nonconstant function**
	'''
	xi1,xi2 = -1./np.sqrt(3), 1./np.sqrt(3)
	xi1 = 0.5*(x2-x1)*xi1+0.5*(x1+x2)
	xi2 = 0.5*(x2-x1)*xi2+0.5*(x1+x2)
	w = 0.57735
	return 0.5*(x2-x1)*w*(f(xi1)+f(xi2))

def read_events(filename):
	''' get earthquake event times from file
	'''
	t,m = np.genfromtxt(filename).T

	# truncate to magnitude of completeness
	t = t[np.where(m>=2.5)]
	m = m[np.where(m>=2.5)]

	return t, m

def calibrate_seismicity(ti, dat, event_times, logpclims = [-3,0], logklims = [-5,5], wgt = None):
	''' implements an MCMC calibration routine using likelihood for nonhomogeneous poisson process
	'''
	ndim = 2			# free parameters
	nwalkers = 10		# independent walker chains

	# MCMC object	
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = [ti, dat, event_times, wgt, logpclims, logklims], threads=6)	
	chain_file = 'chain.csv'
	
	# set up initial positions of the walkers [logpc, k]
	pos = np.array([np.array([-1.3, -5])+1e-2*np.random.randn(ndim) for i in range(nwalkers)])

	# records result of MCMC		
	f = open(chain_file,'w')
	f.close()   
	
	nit = 25				# number of iterations between save
	for i in range(nit):
		pos,prob,state=sampler.run_mcmc(pos,20)
		f = open(chain_file, "w")
		nk,nit,ndim=sampler.chain.shape
		for k in range(nk):
			for i in range(nit):
				f.write("{:d} {:d} ".format(k, i))
				for j in range(ndim):
					f.write("{:15.7f} ".format(sampler.chain[k,i,j]))
				f.write("{:15.7f}\n".format(sampler.lnprobability[k,i]))
		f.close()   

def plot_seismicity_models(ti, dat, event_times, event_mags, save='calibration.png'):
	''' plot the results of calibration
	'''
	CHAIN = 'chain.csv'
	BURNIN = 2

	# corner plot of posteriors
	labels = ['log10(pc)','log10(k)']
	chain = np.genfromtxt(CHAIN)
	nw,it,pc,k,LLK = chain.T
	chain = chain[:,2:-1]
	npar = chain.shape[1]-1
	fig = corner.corner(chain, labels=labels, bins=30, quantiles=[0.05,0.5,0.95])
	fig.set_size_inches([8,8])
	fig.savefig(save.split('.')[0]+'_corner.png', dpi=300)

	# best fitting model
	f,ax = plt.subplots(1,1,figsize=(8,8))
	ibest = np.argmax(chain[:,-1])
	pcbest,kbest = chain[ibest,:]	
	n = simulate_seismicity(ti, dat, pc = 10**pcbest, k = 10**kbest, wgt = None)
	LLK = compute_likelihood(ti, n, event_times)
	ax.plot(ti,n,'k-',label='best model')
	ax.set_title('best model: pc = {:3.2f} MPa, k = {:3.2e}, LLK={:3.2f}'.format(pcbest,kbest,LLK))

	# show magnitudes
	ax2 = ax.twinx()
	ax2.set_ylim([1,5])
	for ti,mi in zip(event_times, event_mags):
		ax2.plot([ti,ti], [1,mi],'r-')
		ax2.plot(ti,mi,'rd')

	ax.set_xlabel('time')
	ax2.set_ylabel('magnitude')
	ax.set_ylabel('seismicity rate')
	ax.legend()
	plt.savefig(save, dpi=400)

def check_par_bounds(logpclims, logklims, pc, k):
	''' check out of bounds for calibration
	'''
	return ((logklims[0]<k<logklims[-1]) and (logpclims[0]<=pc<logpclims[-1]))

def compute_likelihood(ti, n, event_times):
	''' nonhomogeneous poisson process likelihood given rate parameter N and 
		earthquake event times
	'''
	# interpolate simulated seismicity rate to event times
	ri = np.interp(event_times, ti, n)
	
	# truncate integral at final recorded event
	i1 = np.argmin(abs(ti-event_times[-1]))

	# likelihood
	llk = np.sum(np.log(ri))-np.sum(n[:i1])*(ti[1]-ti[0])
	
	# NaN check
	if llk != llk:
		llk = -1.e-32
		
	return llk

def lnprob(pars, ti, dat, event_times, wgt, logpclims, logklims):
	''' likelihood function for emcee
	'''
	pc,k = pars
		# check parameter bounds
	inBounds = check_par_bounds(logpclims, logklims, pc, k)
	if not inBounds:
		return -1.e32
		
	n = simulate_seismicity(ti, dat, 10**pc, 10**k, wgt)
	
	LLK = compute_likelihood(ti, n, event_times)

	return LLK

if __name__ == "__main__":

	# flags
	plotDataSlices = False
	runCalibration = False
	plotCalibration = True

	# load data
	dat = Fault('tercio_vw_time_faultpress.txt')
	event_times, event_mags = read_events('newreloc_041418_jennyvel_tercio_allevents.txt')
	
	if plotDataSlices:
		# generate 2D plots of pressure at halfway points (v,w,t)
		dat.plot(v=np.mean(dat.vu), save='dP_vmid.png')
		dat.plot(w=np.mean(dat.wu), save='dP_wmid.png')
		for ti in np.linspace(np.min(dat.tu),np.max(dat.tu),11)[1:]:
			dat.plot(t=ti, save='dP_ti={:8.7e}.png'.format(ti), clim = [0,0.3])

	# simple plot of two seismicity rates for different parameters
	ti = np.linspace(np.min(dat.tu), np.max(dat.tu), 101)
	n = simulate_seismicity(ti, dat, pc = 0.1, k = 1, wgt = None)

	f,ax = plt.subplots(1,1)
	ax.plot(ti,n,'k-',label='pc=0.1')
	
	n = simulate_seismicity(ti, dat, pc = 0.05, k = 1, wgt = None)
	ax.plot(ti,n,'r-',label='pc=0.05')

	ax.legend()
	ax.set_xlabel('time')
	ax.set_ylabel('relative seismicity rate')
	plt.savefig('sample_seismicity.png', dpi = 400)

	# run emcee calibration - this step takes time
	if runCalibration:
		calibrate_seismicity(ti, dat, event_times, logpclims = [-3,1], logklims = [-20,20], wgt = None)

	# plot the results of emcee calibration
	if plotCalibration:
		plot_seismicity_models(ti, dat, event_times, event_mags, save='calibration.png')
	