
# Model of Western Oklahoma seimsicity 
# ------------------------------------
#
# Supplementary material to Dempsey and Riffault (2018) "Response of induced seismicity to injection rate reduction: models of delay, decay, quiescence, recovery and Oklahoma"
#
# Authors
# -------
# David Dempsey, Jeremy Riffault
#
# Instructions
# ------------
# To run this file, user will need to install Python 2.7, supporting packages, have access to an executable copy of fehm (fehm.lanl.gov) and the Python library PyFEHM (pyfehm.lanl.gov).

# imports
# -------
	# classes for FEHM reservoir simulation
from fdata import*
	# classes to compute induced seismicity from pressure
from inducedseismicity import InducedSeismicity
from scipy.stats import gumbel_r, poisson


# simulation parameters
# ---------------------
	# injection
r_inj = 50.e3			   # radius of injection zone [m]
P0 = 30.					# reference pressure [MPa]
T0 = 100.				   # reference temperature [degC]
scenario = 'current'		# future scenarion ['current', 'mandate', 'half']
	# aquifer
log_k = -11.33			  
k = 10**log_k			   # permeability [m^2]
phi = 0.1				   # porosity
compressibility = 2.e-11	# 
h = 5.e2					# thickness [m]
	# basement
log_kb = -15.47
kb = 10**log_kb			 # permeability [m^2]
phi_b = 0.01				# porosity
hb = 6.e3				   # thickness [m]
	# seismicity
pc = 0.023				   # critical triggering pressure [MPa]
ln = 4.11				   
kappa = 2.29				# scale factor [MPa^-1 km^-2]
r_seis = -4.2e3			 # seismogenic depth [m]

# model grid 
# ----------
	# radial coordinates
rmin,rmax = [r_inj/100., 100*r_inj]
rmid = 2*r_inj
r = np.array(list(np.linspace(rmin, rmid, 50))+list(np.logspace(np.log10(rmid), np.log10(rmax), 201))[1:])
i1 = np.argmin(abs(r-r_inj))
r_inj = 0.5*(r[i1]+r[i1+1])
	# vertical coordinates
nz = 21
z = np.linspace(-hb,0,2*nz)[::2]	
z = np.concatenate([z, [-z[-1], h]])
	# simulation object
dat = fdata()
	# make linear grid
dat.grid.make('grid.inp', x=r, y = [-0.5,0.5], z = z)
	# modify to radial grid (wedge)
dtheta = 1./180.*np.pi		# rotation angle, 1 degree
for nd in dat.grid.nodelist:
	xi,yi,zi = nd._position
	if yi>0:
		nd._position = [xi, xi*np.sin(dtheta), zi]
	else:
		nd._position = [xi, -xi*np.sin(dtheta), zi]
	# write grid to file
dat.grid.write('grid.inp')  

# assign material parameters
# --------------------------
	# default (overwritten by aquifer and basement)
dat.zone[0].permeability = k
dat.zone[0].density = 2500.
dat.zone[0].porosity = phi
dat.zone[0].specific_heat = 1.e3
dat.zone[0].conductivity = 2.5
dat.zone[0].Pi = P0
dat.zone[0].Ti = T0
	# aquifer
dat.new_zone(2, 'reservoir',rect=[[0,-1e8,-0.1],[1e8,1e8,1e8]])
dat.zone[2].permeability = k
dat.zone[2].porosity = phi
dat.zone[2].specific_heat = 1.e3
dat.zone[2].conductivity = 2.5
ppor = fmodel('ppor',index=1, zonelist = 2, param=(('compressibility',compressibility)))
dat.add(ppor)
	# basement
dat.new_zone(3, 'basement', rect=[[0,-1e8,-1.e8],[1e8,1e8,-0.1]])
dat.zone[3].permeability = [1.e-20, 1.e-20, kb]
dat.zone[3].porosity = phi_b
dat.zone[3].specific_heat = 1.e3
dat.zone[3].conductivity = 2.5

# injection source
# ----------------
	# zone
dat.new_zone(1, 'injection', rect=[[0,-1e8,-0.1],[r_inj,1e8,1e8]])
	# read rate from file
t,q = np.genfromtxt(r'WO_inj_all.txt',delimiter=',',skip_header=1).T
	# divide by 180, because we are modeling a 2 degree wedge
q = -q/180.
	# offset time to zero, convert to days
t = (t-t[0])*365.25
	# modify for future scneario
t = np.array(list(t) + [t[-1]+1.,])
if scenario == 'current':
	q = np.array(list(q) + [q[-1],])
elif scenario == 'half':
	q = np.array(list(q) + [q[-1]/2.,])
elif scenario == 'mandate':
	qmandate = -4.e6*1.e3*12./(24*3600*365.25)/180
	q = np.array(list(q) + [qmandate,])
	# injection temperature (isothermal problem)
dat.add(fboun(type='ti', zone=1, times=list(t), variable=[['dsw',]+list(q), ['ft',]+len(q)*[T0,]]))

# model output
# ------------
	# variables
dat.hist.variables.append(['pressure water'])
	# frequency
dat.hist.timestep_interval=1
	# location
dat.hist.nodelist = [nd for nd in dat.grid.nodelist if nd.position[1]<0]

# other parameters
# ----------------
	# turn off gravity effects
dat.ctrl['gravity_direction_AGRAV']=0
	# timestepping
dat.ti = 0.							 # initial time
dat.tf = 22.*365.25					 # final time [days]
dat.dtmax = dat.tf/100.				 # max timestep [days]
dat.dtmin = 1.e-10*dat.tf			   # min timestep [days]
dat.dtn = 10000						 # max number timesteps 

# run the simulation
# ------------------
exe = 'fehm.exe'						# CHANGE to path of your FEHM executable
dat.run(input='ok_model.dat', exe=exe, verbose=False)
		
# process output
# --------------
	# read history for pressure and time data 
h = fhistory("*presWAT_his.dat")
t = h.times
	# interpolator fails for duplicated time values (these occur in some FEHM runs when the simulator
	# takes a very small time step below the output resolution)
	# this bug is fixed by incrementing the duplicated time value an small amount
for j in range(len(t)-1):
	if t[j] == t[j-1]:
		t[j]+= 1.e-6

nds = h.nodes				   # all nodes
tol = 1.e-2
mu = -4.45e3				   # Gumbel dist. parameter [m]
beta = 0.69e3				   # Gumbel dist. parameter [m]
	# loop over depths and extract pressure, weight at each
zs = np.sort(z)[::-1]
dz = abs(zs[-2]-zs[-1]) 		# layer height
ps = []; ws = []
for i in range(1, len(zs)):
	# get nodes at appropriate depth
	ndsz = [nd for nd in nds if abs(dat.grid.node[nd].position[2]-zs[i])<tol]
	ndsz = sorted(ndsz, key = lambda x: dat.grid.node[x].position[0])
	r = np.array([dat.grid.node[nd].position[0] for nd in ndsz])
	
	# compute weight with Gumbel dist. No need to weight with dz as vertical grid is linearly spaced now
	ws.append(abs(gumbel_r().cdf((zs[i]-dz/2-mu)/beta)-gumbel_r().cdf((zs[i]+dz/2-mu)/beta)))
	
	# compute pressure differences from initial time
	ps.append(np.array([h['P'][nd] - P0 for nd in ndsz]))

# compute seismicity rate
# -----------------------
	# convert units 
t = t/365.25+2004			# decimal years
r = r/1.e3					# km
ti = np.linspace(2004,2025,1001)   	# interpolation times
n = 0.*ti
	# loop over layers and compute seismicity contribution of each
for p, w in zip(ps, ws):
	# format FEHM data
	sim = {'r':r,'t':t,'p':p}
	# create object
	iso = InducedSeismicity(f=0.5, dim=2.5, sim = sim, pcrit = pc)
	# compute and append seismicity at depth
	n += iso.nt(ti,rmax)*w
	
	# scale integral to give a seismicity rate (per year)
n *= kappa
	# add Oklahoma background rate, scaled for area of WO
n += 1.16*13/181.

# compute log likelihood
# ----------------------
	# load declustered EQ observations
times = np.genfromtxt('EQ_Oklahoma_WO_dcl.csv', usecols=0, delimiter=',')[::-1]
	# interpolate simulated rate at earthquake times
ri = np.interp(times, ti, n)
	# find index of last event
i1 = np.argmin(abs(ti-times[-1]))
	# compute cumulative rate to last event (rectangular rule)
crt = np.sum(n[:i1])*(ti[1]-ti[0])
	# compute likelihood
LLK = np.sum(np.log(ri))-crt
	  
# plot seismicity rate
# --------------------
f, ax = plt.subplots(1,1)
f.set_size_inches([5,5])
tbins = np.linspace(2012,2018,25)   # bins for observed seismicity rate
h,e = np.histogram(times,tbins)        
rt = h/np.diff(e)/12
tmid = 0.5*(e[:-1]+e[1:])
ax.plot(tmid, rt, 'ko', mfc='w', mew=1.5, label='observed, declustered') 
i1 = np.argmin(abs(ti-2018))
ax.plot(ti[:i1], n[:i1]/12,'k-', label='simulated, LLK={:3.2e}'.format(LLK))
# poisson envelopes
rts_poisson_05 = np.array([poisson.ppf(0.05, ni) for ni in n[:i1]/12])
rts_poisson_95 = np.array([poisson.ppf(0.95, ni) for ni in n[:i1]/12])
ax.fill_between(ti[:i1], rts_poisson_05, rts_poisson_95, color='k', alpha=.5)
	
colors = [[217,95,2],[117,112,179],[27,158,119],]
colors = [np.array(color)/255. for color in colors]
color1,color2,color3 = colors
if scenario == 'current':
	color = color1
elif scenario == 'half':
	color = color2
elif scenario == 'mandate':
	color = color3
ax.plot(ti[i1:], n[i1:]/12,'-', color=color, label='scenario: {}'.format(scenario))
# poisson envelopes
rts_poisson_05 = np.array([poisson.ppf(0.05, ni) for ni in n[i1:]/12])
rts_poisson_95 = np.array([poisson.ppf(0.95, ni) for ni in n[i1:]/12])
ax.fill_between(ti[i1:], rts_poisson_05, rts_poisson_95, color=color, alpha=.5)
ax.legend()
ax.set_xlim([2012, 2025])
ax.set_xlabel('time')
ax.set_ylabel('M$\geq$3 earthquake per month')
plt.show()
	
		
 
 
 
 
