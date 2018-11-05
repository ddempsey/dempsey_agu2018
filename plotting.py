
# plotting.py
# -----------
#
# Plotting functions that support Jupyter Notebook. Supplementary material to Dempsey and Riffault (2018) "Response of induced seismicity to injection rate reduction: models of delay, decay, quiescence, recovery and Oklahoma"
#
# Authors
# -------
# David Dempsey, Jeremy Riffault

import numpy as np
from inducedseismicity import InducedSeismicity
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.special import expi, jv, yv, jn_zeros
from scipy.stats import poisson
from datetime import datetime
from ipywidgets import interact
import csv

# functions for notebook plotting
text_size = 12
colors = [[217,95,2],[117,112,179],[27,158,119],]
colors = [np.array(color)/255. for color in colors]
color1,color2,color3 = colors

# Bessel functions
def j0(x): return jv(0,x)
def j1(x): return jv(1,x)
def y0(x): return yv(0,x)
def y1(x): return yv(1,x)
    
# helper functions for bessel integration
def psi(t): return t*np.tanh(np.pi/2*np.sinh(t))
def d_psi(t):
    q = (np.pi*t*np.cosh(t)+np.sinh(np.pi*np.sinh(t)))/(1+np.cosh(np.pi*np.sinh(t)))
    q[np.where(q != q)] = 1
    return q
def jquad(func, v=0, h=1.e-3, N=200, **kwargs):
    ''' Numerically evaluate integrals of the following form
             /+inf
        I = | f(x).*besselj(v,x)dx 
            /0
        
        Parameters
        ----------
        func : function handle 
            Function to integrate, f(x).
        v : int 
            Bessel function order.
        h : float
            Step size.
        N : int
            Number of nodes.
        
        Returns
        -------
        I : float
            Approximation of integral.
        
        Notes
        -----
        Based on Ogata's method,
        [1] H. Ogata, A Numerical Integration Formula Based on the Bessel Functions, 
         Publications of the Research Institute for Mathematical Sciences, vol. 41, no. 4, pp. 949970, 2005.
        
        Python reimplementation of MATLAB code by AmirNader Askarpour (2011). Original source at https://www.mathworks.com/matlabcentral/fileexchange/33929-quadrature-rule-for-evaluating-integrals-containing-bessel-functions
    '''
    
    # calculating zeros of Bessel function
    xi_vk = jn_zeros(v,N)/np.pi

    # weights
    w_vk = yv(v,np.pi*xi_vk)/jv(v+1,np.pi*xi_vk)

    # quadrature
    S = np.pi*w_vk*func(np.pi/h*psi(h*xi_vk),**kwargs)*jv(v,np.pi/h*psi(h*xi_vk))*d_psi(h*xi_vk)

    return np.sum(S);
def iu1(u, a, r, t, f):
    fx = 16./np.pi**2*(1.-np.exp(-(a*u)**2*t))*j0(u*r)/(u**4*(j1(u)*y0(u)-j0(u)*y1(u))**2)
    if t>1:
        fx += 16./np.pi**2*(-f+f*np.exp(-(a*u)**2*(t-1.)))*j0(u*r)/(u**4*(j1(u)*y0(u)-j0(u)*y1(u))**2)
    return fx
def iu2(u, a, r, t, f):
    fx = 8./np.pi*(1.-np.exp(-(a*u)**2*t))*j0(u*r)/(u**3*(j1(u)*y0(u)-j0(u)*y1(u)))
    if t>1:
        fx += 8./np.pi*(-f+f*np.exp(-(a*u)**2*(t-1.)))*j0(u*r)/(u**3*(j1(u)*y0(u)-j0(u)*y1(u)))
    return fx   
def get_OKWO_eqs():
    te,me = [],[]
    t0 = datetime.strptime('20090101', '%Y%m%d')           # reference time 1 Jan 1991
    fl = 'EQ_Oklahoma_WO_dcl.csv'
    with open(fl, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')  
        for row in reader:
            if len(row) == 0: continue
            te.append(float(row[0]))
            me.append(float(row[3]))
    te = np.array(te)
    me = np.array(me)
    inds = np.where(me>0.999)
    te = te[inds]
    me = me[inds]
    return te, me

# plotting functions
def draw_pressure(fig, ax, iso, show, advance):
    '''
    '''
    fig.set_size_inches([12,6])
    ts = np.logspace(-1,2,201)
    rs = np.logspace(-1,1,201)
    pbf = iso.prt(iso.tstar, iso.rstar)
    
    if show == 'distance':
        # contours
        list_r = [.2, .32, iso.rstar, .65, 1., 2., 4.][:advance]
        
        for i_r, r in enumerate(list_r):
            # compute
            ps = [iso.prt(t,r) for t in ts]
            # colours
            if r < iso.rstar: c = color2   
            elif r == iso.rstar: c = 'k'
            else: c = color1
            # plot
            ax.plot(ts, ps, c=c)
        # upkeep
        xl = 'dimensionless time, $t$'
        yl = 'dimensionless pressure, $p$'
        ys = 'linear'
        xlim = [0.1,100]
        ylim = [0,5]
        xbf = iso.tstar
        ybf = pbf
        ax.plot([], [], color=color2, label='$r<r^*$')
        ax.plot([], [], 'k', label='$r=r^*$')
        ax.plot([], [], color=color1, label='$r>r^*$')
        leg = ax.legend(loc='upper left')  
    elif show == 'time':
        # contours
        list_t = [.02, .1, .28, 1., iso.tstar, iso.gr(rs[0])[1][0], 5., 10., 20.][:advance]
        for i_t, t in enumerate(list_t):
            # compute
            ps = [iso.prt(t,r) for r in rs]
            # colours
            if t == 1: ls = '--'
            else: ls = '-'
            if t in [1, iso.tstar]: lw = 1.
            else: lw = .5
            if t < iso.tstar: c = color2
            elif t == iso.tstar: c = 'k'
            else: c = color1
            # plot
            ax.plot(rs, ps, lw=lw, c=c, ls=ls)
        # upkeep
        xl = 'dimensionless radius, $r$'
        yl = 'dimensionless pressure, $p$'
        ys = 'linear'
        xlim = [0.1,10]
        ylim = [0,5]
        xbf = iso.rstar
        ybf = pbf
        ax.plot([], [], color=color2, lw = .4,  label='$t<t^*$')
        ax.plot([], [], color=color2, lw = 1,  ls='--', label='$t=1$')
        ax.plot([], [], 'k', lw = 1, label='$t=t^*$')
        ax.plot([], [], color=color1, lw = .4,  label='$t>t^*$')
        ax.legend(loc='upper right')
    elif show == 'pressure':
        # contours
        list_P = [.5, 1., pbf, 2., 3., 4.][:advance]
        for i_P, P in enumerate(list_P):
            # compute
            r_i = [iso.rp(t, P) for t in ts]
            # colours
            if P < pbf: c = color2   
            elif P == pbf: c = 'k'
            else: c = color1
            # plot
            ax.plot(ts, r_i, c=c)
        # upkeep
        xl = 'dimensionless time, $t$'
        yl = 'dimensionless radius, $r$'
        ys = 'log'
        xlim = [0.1,100]
        ylim = [0.1,10]
        xbf = iso.tstar
        ybf = iso.rstar
        ax.plot([], [], color=color2, label='$p<p^*$')
        ax.plot([], [], 'k', label='$p=p^*$')
        ax.plot([], [], color=color1, label='$p>p^*$')
        leg = ax.legend(loc='upper center')
    else:
        raise ValueError
        
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    ax.set_yscale(ys)
    if show == 'pressure':
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.plot([xbf,xbf,xlim[0]],[ylim[0],ybf,ybf],'k:')
    if show != 'time':
        ax.plot([1,1],ylim,'k--')
def draw_pressure_distributed(fig, ax, loga, f, advance):
    '''
        iso - InducedSeismicity object
    '''
    fig.set_size_inches([12,6])
    rs = np.linspace(0., 4., 101)
    ts = np.logspace(-1,1,9)[:advance]
    a = 10**loga
    for t in ts:
        fi = f
        if t<1:
            fi = 0.
        ps = [jquad(iu1,a=a,r=ri,t=t,f=fi,v=1) if ri<1. else jquad(iu2,a=a,r=ri,t=t,f=fi,v=1) for ri in rs]
    
        if t < 1.:
            ls = '--'
            c = color1
        elif t == 1.:
            ls = '-'
            c = color1
        elif t>1:
            ls = '-'
            c = color2
        ax.plot(rs,ps,c=c,ls=ls)
    
    # plot upkeep
    ax.set_xlabel('radius, $r$')
    ax.set_ylabel('pressure, $p$')
    ax.set_xlim([0,4])
    ylim = ax.get_ylim()
    ax.set_ylim([0,ylim[-1]])
def draw_seismicity(fig, axs, iso):
    # unpack axes
    (ax1,ax2) = axs
    
    # resize figure
    fig.set_size_inches(12,10)

    # create plot vectors
    t = np.logspace(-1.,2.,201)
    
    # SUBPLOT 1
    
    # plot isobars
    p = np.array([.1, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]) 
    rv = np.logspace(-1,.3,101)
    tv = np.array(t)
    tt,rr = np.meshgrid(tv,rv)
    zz = 0.*tt
    for i in range(rr.shape[0]):
        zz[i,:] = iso.prt(tt[i,:],rr[i,0])
    CS = ax1.contour(tt, rr, zz, levels = p, colors = ('k'), alpha=.5)
    ax1.plot([],[],'k', alpha=.5, label='isobar')
    
    # plot rcrit
    ax1.plot(tv, iso.rp(tv,iso.pc), 'k-',label='$p=p_{c}$')
    
    # plot wellhead pressure if in 1D
    if iso.d == 1:
        ax1.plot(tv, 2*np.sqrt(t/np.pi), 'k--',label='$p=p_{well}$')
    
    # plot g(t), h(t) functions
    tg = np.logspace(np.log10(1+1.e-5), np.log10(1/(1-iso.f**(2./iso.d))-1.e-5), 101)
    ax1.plot(tg, iso.gt(tg), 'k:',label='$g(t)$')
    
    th = np.logspace(np.log10(iso.tstar+1.e-1), np.log10(tv[-1]), 101)
    ax1.plot(th, iso.ht(th), 'k-.',label='$h(t)$')
    ax1.plot(iso.tstar, iso.rstar, 'k^', mew = 1.5, ms=10, mfc='w', mec='k')
    ax1.text(iso.tstar, iso.rstar*1.05, '$[t^*,r^*]$', ha='center', va='bottom', fontsize=text_size)
    
    # plot zones
    tg1 = np.logspace(np.log10(1+1.e-5), np.log10(iso.tstar), 101)
    tg2 = np.logspace(np.log10(iso.tstar), np.log10(1/(1-iso.f**(2./iso.d))-1.e-5), 101)
        
    ax1.fill([1.,*tg,1/(1-iso.f**(2./iso.d))],[rv[0],*iso.gt(tg),rv[0]],color=color2,label='$p$ decrease')
    ax1.fill([*th,tv[-1],*tg2[::-1]],[*iso.ht(th),rv[0],*iso.gt(tg2[::-1])],color=color1,label='Kaiser effect')
    
    if iso.pc < iso.pstar:
        ax1.fill([tv[0],1.,*tg1,*th,*tv[::-1]],[rv[0],rv[0],*iso.gt(tg1),*iso.ht(th),*iso.rp(tv[::-1],iso.pc)],color=color3,label='seismicity')
    else:
        
        # plot a and b intersections
        ta, ra = iso.gp(iso.pc)
        ax1.plot(ta, ra, 'kx', mew = 1.5)
        ax1.text(ta, ra, '$t_g$', ha='right', va='bottom', fontsize=text_size)
        tb, rb = iso.hp(iso.pc)
        ax1.plot(tb, rb, 'kx', mew = 1.5)
        if tb<tv[-1]:
            ax1.text(tb, rb, '$t_h$', ha='left', va='bottom', fontsize=text_size)
        
        # fill
        tga = np.logspace(np.log10(1+1.e-5), np.log10(ta), 101)
        tra = np.logspace(np.log10(tv[0]), np.log10(ta), 101)
        
        ax1.fill([tv[0],*tga,*tra[::-1]],[rv[0],*iso.gt(tga),*iso.rp(tra[::-1], iso.pc)],color=color3,label='seismicity')
        thb = np.logspace(np.log10(tb), np.log10(tv[-1]), 101)
        ax1.fill([*thb,*thb[::-1]],[*iso.ht(thb),*iso.rp(thb[::-1], iso.pc)],color=color3)
        
    # annotation contours
    plabel = [pi for pi in p if pi>iso.prt(iso.tstar,iso.rstar)] 
    t_locations = np.logspace(np.log10(iso.tstar+2.), np.log10(th[-1]), len(plabel))
    manual_locations = [(ti,iso.rp(ti,pi)) for pi, ti in zip(plabel, t_locations)]
    try:
        ax1.clabel(CS, inline=1, fontsize=text_size, manual=manual_locations,fmt='%1.1f')
    except:
        pass
    
    # SUBPLOT 2
    
    # plot normed seismicity rate variation with pcrit
    t_text = 60.
    n = iso.nt(t)
    ax2.plot(t, n, 'k-',label='$n(t)$')
    ax2.axvline(1, c='k', ls='--')
    
    # plot upkeep
    ax1.set_yscale('log')
    ax1.set_ylabel('dimensionless radius, $r$',fontsize=text_size)
    ax1.set_ylim([rv[0],rv[-1]])
    ax1.set_xlim([tv[0],tv[-1]])
    ax1.set_yticks([.1,1,2])
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.legend(fontsize = text_size, loc='upper right')
    
    ax2.set_ylim(0,1.2)
    ax2.set_ylabel('normalized seismicity rate, $N$',fontsize=text_size)
    ax2.legend(fontsize = text_size, loc='upper right')
    
    for ax in [ax1,ax2]:
        ax.set_xscale('log')
        ax.set_xlim(t[0],t[-1])
        ax.tick_params(labelsize=text_size)
        ax.set_xticks([])
    ax2.set_xlabel('dimensionless time, $t$',fontsize=text_size)
    ax2.set_xticks([1,10,100])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    # create plot vectors
    t = np.logspace(-1.,2.,41)
def draw_sensitivity(fig, ax, iso):
    # resize figure
    fig.set_size_inches(10,6)

    # create plot vectors
    t = np.logspace(-1.,2.,201)
    
    # plot normed seismicity rate variation with pcrit
    n = iso.nt(t)
    ax.plot(t, n, 'k-', label = '$N$')
    if iso.pc >1.e-6:
        iso.pc = 0.
        n = iso.nt(t)
        ax.plot(t, n, 'k:', label = '$N(p_c=0)$')
    
    
    ax.set_ylim(0,1.2)
    ax.set_ylabel('normalized seismicity rate, $N$',fontsize=text_size)
    ax.legend(fontsize = text_size, loc='upper right')
    
    ax.set_xscale('log')
    ax.set_xlim(t[0],t[-1])
    ax.tick_params(labelsize=text_size)
    ax.set_xticks([])
    ax.set_xlabel('time',fontsize=text_size)
    ax.set_xticks([1,10,100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
def draw_oklahoma(fig, ax, x, saves):
 
    #### load variables
    labels = [
        'calibrated models',
        'future inj. at Jan \'18 rate',
        '50% Jan \'18 rate',
        'mandated maximum'
        ]
    colors_loc = ['k']+colors
    
    # read saves
    t1 = int(saves[1,0])                                # load index time transition calibration to forecast
    t = saves[2:np.where(saves[:,0]<1.)[0][0],0]        # load time
    t2 = len(t[t1:])                                    # length time vector forecast
    tj = [t[:t1+1]]+[t[t1:]]*3                          # list time vectors for 4 different plots
    ts = [range(t1+1)]+[range(t1+1+i*t2,t1+1+(i+1)*t2) for i in range(3)]   # list range indexes for 4 different plots    
    i_n = np.where([sum(abs(x-x_i))<1.e-2 for x_i in saves[1:5,1:].T])[0][0]    # read index parameters
    rts = saves[9:, i_n+1]            # load seismicity rates
    rts_poisson_05 = np.array([poisson.ppf(0.05, r) for r in rts])   # calculate poisson distribution  5% envelope
    rts_poisson_95 = np.array([poisson.ppf(0.95, r) for r in rts])   # calculate poisson distribution 95% envelope
    LLK = saves[0, i_n+1]             # load log likelihood value

    # start plotting
    fig.set_size_inches(10,6)   # resize figure
    for j in range(4): 
        ax.plot(tj[j], rts[ts[j]], c=colors_loc[j], label=labels[j])     # plot seismicity rate
        ax.fill_between(tj[j], rts_poisson_05[ts[j]], rts_poisson_95[ts[j]], color=colors_loc[j], alpha=.5)     # plot Poisson enveloppe

    # plot Oklahoma recorded EQ
    te,me = get_OKWO_eqs()            # read list of EQ
    tbins = np.linspace(2012,2018,25)   # bins for observed seismicity rate
    h,e = np.histogram(te,tbins)        
    rt = h/np.diff(e)/12
    tmid = 0.5*(e[:-1]+e[1:])
    ax.plot(tmid, rt, 'ko', mfc='w', mew=1.5, label='observed, declustered') 
            
    # upkeep
    ax.set_xlim(2012, 2025)
    ax.set_xlabel('time',fontsize=text_size)
    ax.set_ylim(0, 50)
    ax.set_ylabel('Monthly number of earthquakes ('+r'$M \geq 3$)',fontsize=text_size)
    ax.legend(fontsize=text_size, loc='upper right')
    ax.text(.8, .6, 'LLK='+str(int(LLK)), ha='center', transform=ax.transAxes)
def setup_oklahoma():
    fig = plt.figure()
    fig.set_size_inches(10,6)
    x0,y0,dx,dy,dx2,dy2 = [0.1,0.1,0.8,0.8,0.4,0.4]
    ax = plt.axes([x0,y0,dx,dy])
    ax2 = plt.axes([x0+dx-dx2,y0+dy-dy2,dx2,dy2])
    return fig,[ax,ax2]

