
import numpy as np
from inducedseismicity import *
from copy import copy
from matplotlib import pyplot as plt

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
		v,w,t,p = np.genfromtxt(self.filename).T
		self._grid(v,w,t,p)
		
	def _grid(self,v,w,t,p):
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
	''' compute bulk induced seismicity for fault

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
		n += iso.nt(ti)*w

	return n*k
		
def gauss_int(f,x1,x2):
	''' estimate integratal of function F between X1 and X2 using Gaussian quadrature
	'''
	xi1,xi2 = -1./np.sqrt(3), 1./np.sqrt(3)
	xi1 = 0.5*(x2-x1)*xi1+0.5*(x1+x2)
	xi2 = 0.5*(x2-x1)*xi2+0.5*(x1+x2)
	w = 0.57735
	return 0.5*(x2-x1)*w*(f(xi1)+f(xi2))


if __name__ == "__main__":

	# flags
	plotting = False

	# load data
	dat = Fault('tercio_vw_time_faultpress.txt')
	
	if plotting:
		# generate 2D plots of pressure at halfway points (v,w,t)
		dat.plot(v=np.mean(dat.vu), save='dP_vmid.png')
		dat.plot(w=np.mean(dat.wu), save='dP_wmid.png')
		for ti in np.linspace(np.min(dat.tu),np.max(dat.tu),11)[1:]:
			dat.plot(t=ti, save='dP_ti={:8.7e}.png'.format(ti), clim = [0,0.3])

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
	