#-----------------------------------------------------------------------------80
# NOTE: DOCUMENTATION FOR THIS FILE IS STILL IN PROGRESS

import cupy as cp
import freud
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Polygon
import matplotlib.animation
from matplotlib.animation import FuncAnimation 
from IPython.display import display,HTML
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from IPython.display import clear_output
from kmcsim.utils import time_execution


# Definitions for various fitting functions
pwrlaw = lambda t, c, K, a: (K + c) * (t**a)
pwrlawD = lambda t, D, a: 4 * D * (t**a)
r2 = lambda y, ypred: np.abs(1 - np.sum((y - ypred)**2) / np.sum((y - np.mean(y))**2))
rmse = lambda y, ypred: np.sqrt(np.sum((y - ypred)**2) / len(y))
constfunc = lambda x, c: x/x*c

pwrlawdecay = lambda x, m,a: m * (x)**(-a)
pwrlawdecay2 = lambda x,m,d,a: m * (x)**(-a)+d

expdecay = lambda x, m,a: m * np.exp(-(x) / a)
def function(a, x, n):
    term1 = a ** (-n)
    term2 = -n * a ** (-n - 1)
    term3 = 1/2 * n * (n + 1) * x ** 2 * a ** (-n - 2)
    term4 = -1/6 * x ** 3 * n * (n + 1) * (n + 2) * a ** (-n - 3)
    term5 = 1/24 * n * (n + 1) * (n + 2) * (n + 3) * x ** 4 * a ** (-n - 4)
    term6 = -1/120 * x ** 5 * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * a ** (-n - 5)
  
    return term1 + term2 + term3 + term4 + term5 + term6


def remove_pbc(sim):
    """
    Removes periodic boundary conditions from trajectories and sets every initial position to the origin.
    """
    # Dimensions corresponding to MC step and molecule are swapped to make for easier indexing later
    xydata = cp.array(sim.mxy)
    mxy = xydata.transpose(0,2,1,3)
    dims = cp.array(sim.dims)
    
    # Compute the vector displacements (under PBC) between each MC step
    dxy = cp.diff(mxy, axis=2) + dims/2.
    dxy = (dxy - cp.floor(dxy/dims) * dims) - dims/2.

    # Sequentially add each vector displacement its previous coordinate to get true coordinate outside of box.
    for step in range(1, mxy.shape[2]):
        mxy[:,:,step,:] = mxy[:,:,step-1,:] + dxy[:,:,step-1,:]
    
    # Swap the dimensions back to their original order
    mxy = mxy.transpose(0,2,1,3)
    
    # Centers every trajectory at the origin
    mxy = mxy - mxy[:,0,None,:,:]
    
    return mxy.get()


def get_msdfits(x, y, start, end):
    """
    Returns the fit for the MSD data.
    """
    popt, _ = curve_fit(pwrlaw, x[start:end], y[start:end])
    c, K, alpha = popt
    popt, _ = curve_fit(pwrlawD, x[start:end], y[start:end])
    D = popt[0]
    
    return [x[start:end], int(len(x[start:end])/2), c, K, alpha, np.log(D)]


def get_cffits(x, y, fit_id,sigma=None,r2fit=1,pwrlawconstant=False):
    popt_constfunc = curve_fit(constfunc, x, y,sigma=sigma,maxfev=3000, bounds=(-np.amin(x), np.inf))[0]
    if pwrlawconstant:
        pwrlaw=pwrlawdecay2
    else:
        pwrlaw=pwrlawdecay

    popt_pwrlawdecay = curve_fit(pwrlaw, x, y,sigma=sigma,maxfev=3000, bounds=(-np.amin(x),np.inf))[0]
   
    popt_expdecay = curve_fit(expdecay, x, y,sigma=sigma,maxfev=3000, bounds=(0, np.inf))[0]
    
    fit_ypreds = [constfunc(x, *popt_constfunc), 
                  pwrlaw(x, *popt_pwrlawdecay), 
                  expdecay(x, *popt_expdecay)]
    
    r2_scores = []
    rmse_scores = []
    for ypred in fit_ypreds:
        r2_scores.append(r2(y, ypred))
        rmse_scores.append(rmse(y, ypred))
    
    fit_names = ['constant fit', 'power law fit','exponential fit']
    fit_funcs = [constfunc, pwrlaw, expdecay]
    popts = [popt_constfunc, popt_pwrlawdecay, popt_expdecay]
    annotations = [
        r'$c\simeq ${:.2f}'.format(popts[0][-1]),
        r'$r^{{-\eta}}: $ ($\eta\simeq ${:.2f})'.format(popts[1][-1]),
        r'$\exp(-r/\xi): $($\xi\simeq ${:.2f})'.format(popts[2][-1])
    ]
    if r2fit==0:
        min_id = np.argmin(np.abs(np.array(r2_scores) - 1.))
    if r2fit==1:
        min_id = np.argmin(np.abs(np.array(rmse_scores)))
    if r2fit==2:
        min_id = np.argmin(np.abs(np.array(r2_scores) - 1.)+np.abs(np.array(rmse_scores)))
        
    print(f'popt_constfunc (r2={r2_scores[0]:.5g}, rmse={rmse_scores[0]:.5g}): {popt_constfunc}')
    print(f'popt_pwrlawdecay (r2={r2_scores[1]:.5g}, rmse={rmse_scores[1]:.5g}): {popt_pwrlawdecay}')
    print(f'popt_expdecay (r2={r2_scores[2]:.5g}, rmse={rmse_scores[2]:.5g}): {popt_expdecay}\n')
    
    if fit_id is not None:
        return_fit = [fit_funcs[fit_id], popts[fit_id], annotations[fit_id]]
    else:
        print(f'winner: {fit_names[min_id]}\n')
        return_fit = [fit_funcs[min_id], popts[min_id], annotations[min_id]]        
    
    return return_fit


@time_execution
def get_msds(sim):
    """
    Function that calculates the time-averaged (TA), ensemble-averaged (EA), and time & ensemble-averaged(TEA)
    MSDs of the input data.    
    """
    # Create array of lagtimes from simulation times
    lagtimes = np.arange(sim.times[0].shape[0])
    
    # Remove periodic boundary condtions from data and transform array to 3D for freud library
    xydata = remove_pbc(sim)
    points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)
    
    # Freud requires data centered at the origin
    xdim, ydim = sim.dims
    
    # Define system box object in freud
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    
    # Initialize MSD modules on defined box
    eamsd = freud.msd.MSD(box=box, mode='direct')    
    tamsd = freud.msd.MSD(box=box, mode='window')
    
    # Loop through samples, computing the EA-MSDs and TA-MSDs for each
    eamsds = []
    tamsds = []
    for i, sample in enumerate(points):
        eamsd.compute(sample)
        eamsds.append(eamsd.msd)                
        tamsd.compute(sample)
        tamsds.append(tamsd.msd)
        
    return [lagtimes, np.mean(np.array(eamsds), axis=0), np.array(tamsds), np.mean(np.array(tamsds), axis=0)]


def plot_msds(sim, liney=0.0, textxy=(65., -20.), figsize=(12,6), dpi=72,savename=False,filetype='svg'):
    """
    Plots MSDs of given data.
    """
    msds = get_msds(sim)
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='w', constrained_layout=True, dpi=dpi)    

    ax.plot(msds[0][1:], msds[3][1:], lw=3., c='r', zorder=9, label='TEA-MSD')
    ax.plot(msds[0][1:], msds[1][1:], lw=3., c='blue', zorder=8, label='EA-MSD')        

    ax.plot(msds[0][1:], msds[2][0][1:], lw=0.5, c='grey', alpha=0.5, zorder=7, label='TA-MSDs')
    if sim.n_samples > 1:
        for msd in msds[2]:
            ax.plot(msds[0][1:], msd[1:], lw=0.5, c='grey', alpha=0.5, zorder=7)

    xvals, txtid, cval, Kval, alpha, Dval = get_msdfits(msds[0], msds[3], start=2, end=7)
    ax.plot(xvals, pwrlaw(xvals, cval-liney, Kval, alpha), '--', lw=1.5, c='k', zorder=9)
    ax.annotate(r'$\alpha\simeq ${:.1f} | $\ln(D_{{\alpha}})\simeq ${:.1f}'.format(alpha, Dval), 
                xy=(xvals[txtid], pwrlaw(xvals[txtid], cval-liney, Kval, alpha)), 
                xytext=textxy, textcoords='offset pixels', va='center', ha='center', zorder=9)
    
    title_text = (f'{sim.n_samples} Samples of {sim.n_molecules}-Molecule-Averaged Trajectories '
                  f'({sim.potential_type} interaction, Density = {sim.rho:.4f}, Energy barrier = {sim.eb}K)')
    ax.set_title(title_text, y=1.02)
    ax.set(xlabel=r'time $(\tau)$', ylabel=r'MSD  $({{nm}}^{2})$')
    ax.set(xscale='log', yscale='log', xlim=(msds[0][1], msds[0][-1]), ylim=(msds[3][1], msds[3][-1]))
    ax.legend(shadow=True)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    
    return plt.show()


def plot_boops(sim, sample=0, step=(0,-1), dpi=72,savename=False,filetype='svg'):
    """
    Plots the local bond-orientational order parameter to show 5-7 disclinations and
    orientational order in radians.
    """
    cmap0 = ListedColormap(np.vstack([
        mpl.cm.get_cmap('PRGn_r', 10)(np.arange(10))[np.array([4,3,2,1])],
        mpl.cm.get_cmap('bwr', 3)(np.arange(3)),
        mpl.cm.get_cmap('PuOr_r', 10)(np.arange(10))[np.array([8,7,6,5])]
    ]))
    norm0 = Normalize(vmin=1-0.5, vmax=11+0.5)
    cmap1 = ListedColormap(mpl.cm.get_cmap('hsv_r', 256)(np.linspace(0., 1., 256))[80:])    
    norm1 = Normalize(vmin=0., vmax=np.pi)    
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 13), facecolor='w', constrained_layout=True, dpi=dpi)
    
    xdim, ydim = sim.dims
    xydata = sim.mxy - sim.center_xy
    points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    vor = freud.locality.Voronoi()
    fpsi = freud.order.Hexatic(k=6, weighted=False)
    for i, step_num in enumerate(step):
        mxy = xydata[sample, step_num]
        vor.compute(system=(box, points[sample, step_num]))
        fpsi.compute(system=(box, points[sample, step_num]), neighbors=vor.nlist)
        fpsi_phase = np.abs(np.angle(fpsi.particle_order))
        nsides = np.array([polytope.shape[0] for polytope in vor.polytopes])
        
        patches = []
        for polytope in vor.polytopes:
            poly = Polygon(polytope[:,:2], closed=True, facecolor='r')
            patches.append(poly)
            
        collection0 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap0, norm=norm0, alpha=0.6)
        collection0.set_array(nsides)
        ax0 = ax[0,i].add_collection(collection0)
        
        collection1 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap1, norm=norm1, alpha=0.7)
        collection1.set_array(fpsi_phase)
        ax1 = ax[1,i].add_collection(collection1)
        
        for j in range(2):
            ax[j,i].scatter(mxy[:,0], mxy[:,1], s=1, c='k', zorder=2)
            box.plot(ax=ax[j,i])
        
        step_txt = sim.times[sample].shape[0] if step_num == -1 else step_num
        ax[0,i].set(xlabel=None, xticks=[], title=f'5 and 7-Disclinations (step = {step_txt})')
        ax[1,i].set(title=f'Local Bond-Orientational Order (step = {step_txt})')
        
    for i in range(2):
        ax[i,1].set(ylabel=None, yticks=[])
    
    cax0 = fig.add_axes([ax[0,1].get_position().x1+0.11, 
                         ax[0,1].get_position().y0+0.045, 
                         0.02, 
                         ax[0,1].get_position().height])
    cbar0 = fig.colorbar(ax0, cax=cax0, ticks=np.arange(1, 12))
    cbar0.set_label(label='number of neighbors', labelpad=20., rotation=270)
    cax1 = fig.add_axes([ax[1,1].get_position().x1+0.11, 
                         ax[1,1].get_position().y0-0.02, 
                         0.02, 
                         ax[1,1].get_position().height])
    cbar1 = fig.colorbar(ax1, cax=cax1)
    cbar1.set_ticks(np.arange(0., np.pi+(np.pi/4.), np.pi/4.), labels=['0', 'π/4', 'π/2', '3π/4', 'π'])
    cbar1.set_label(label='orientation in radians', labelpad=10., rotation=270)
    if savename!=False:
        fig.savefig(savename, format=filetype)    
    return plt.show()

def plot_boops2(sim, sample=0, step=(0,-1), dpi=72, size=False,savename=False,filetype='svg'):
    """
    Plots the local bond-orientational order parameter to show 5-7 disclinations and
    orientational order in radians.
    """
    if size==False:
        size=sim.dims[0]
    cmap0 = ListedColormap(np.vstack([
        mpl.cm.get_cmap('PRGn_r', 10)(np.arange(10))[np.array([4,3,2,1])],
        mpl.cm.get_cmap('bwr', 3)(np.arange(3)),
        mpl.cm.get_cmap('PuOr_r', 10)(np.arange(10))[np.array([8,7,6,5])]
    ]))
    norm0 = Normalize(vmin=1-0.5, vmax=11+0.5)
    cmap1 = ListedColormap(mpl.cm.get_cmap('hsv_r', 256)(np.linspace(0., 1., 256))[80:])    
    norm1 = Normalize(vmin=0., vmax=np.pi)    
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 13), facecolor='w', constrained_layout=True, dpi=dpi)
    
    xdim, ydim = sim.dims
    xydata = sim.mxy - sim.center_xy
    points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    vor = freud.locality.Voronoi()
    fpsi = freud.order.Hexatic(k=6, weighted=False)
    for i, step_num in enumerate(step):
        mxy = xydata[sample, step_num]
        vor.compute(system=(box, points[sample, step_num]))
        fpsi.compute(system=(box, points[sample, step_num]), neighbors=vor.nlist)
        fpsi_phase = np.abs(np.angle(fpsi.particle_order))
        nsides = np.array([polytope.shape[0] for polytope in vor.polytopes])
        
        patches = []
        for polytope in vor.polytopes:
            poly = Polygon(polytope[:,:2], closed=True, facecolor='r')
            patches.append(poly)
            
        collection0 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap0, norm=norm0, alpha=0.6)
        collection0.set_array(nsides)
        ax0 = ax[0,i].add_collection(collection0)
        
        collection1 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap1, norm=norm1, alpha=0.7)
        collection1.set_array(fpsi_phase)
        ax1 = ax[1,i].add_collection(collection1)
        
        for j in range(2):
            ax[j,i].scatter(mxy[:,0], mxy[:,1], s=1, c='k', zorder=2)
            box.plot(ax=ax[j,i])
        
        step_txt = sim.times[sample].shape[0] if step_num == -1 else step_num
        ax[0,i].set(xlabel=None, xticks=[], title=f'5 and 7-Disclinations (step = {step_txt})')
        ax[1,i].set(title=f'Local Bond-Orientational Order (step = {step_txt})')
        
    for i in range(2):
        ax[i,1].set(ylabel=None, yticks=[])
    
    cax0 = fig.add_axes([ax[0,1].get_position().x1+0.11, 
                         ax[0,1].get_position().y0+0.045, 
                         0.02, 
                         ax[0,1].get_position().height])
    cbar0 = fig.colorbar(ax0, cax=cax0, ticks=np.arange(1, 12))
    cbar0.set_label(label='number of neighbors', labelpad=20., rotation=270)
    cax1 = fig.add_axes([ax[1,1].get_position().x1+0.11, 
                         ax[1,1].get_position().y0-0.02, 
                         0.02, 
                         ax[1,1].get_position().height])
    cbar1 = fig.colorbar(ax1, cax=cax1)
    cbar1.set_ticks(np.arange(0., np.pi+(np.pi/4.), np.pi/4.), labels=['0', 'π/4', 'π/2', '3π/4', 'π'])
    cbar1.set_label(label='orientation in radians', labelpad=10., rotation=270)
    for i in range (2):
        for j in range (2):
            ax[i,j].set_xlim(-size/2,size/2)
            ax[i,j].set_ylim(-size/2,size/2)
    if savename!=False:
        fig.savefig(savename, format=filetype)    
    return plt.show()
def animate_boops(sim,kind=0,size=False,savename=False,sample=0,prnt=True,Frames=300,step=1):
    fig, ax = plt.subplots(1,1,figsize=(7,6), facecolor='w', constrained_layout=True)
    def animate(frame_number):
        ax.cla()
        if prnt:
            clear_output(wait=True)
            print(frame_number)
        return plot_boops4(sim,sample=sample,step=[frame_number*step],kind=kind,size=size,anim_param=(fig,ax))
    anim=FuncAnimation(fig,animate,frames=Frames,interval=100,save_count=Frames)
    display(HTML(anim.to_jshtml()),)
    if savename!=False:
        anim.save(savename)
def plot_boops4(sim, sample=0, step=[0,-1], dpi=72,savename=False,kind=0,filetype='svg',size=False,anim_param=False):
    """
    Plots the local bond-orientational order parameter to show 5-7 disclinations and
    orientational order in radians.
    """
  
    if size==False:
        size=sim.dims[0]
    if not anim_param:
        fig, ax = plt.subplots(1, np.size(step),figsize=(7*np.size(step),6), facecolor='w', constrained_layout=True, dpi=dpi)
    else:
        fig,ax=anim_param
    if kind==0:
        cmap0 = ListedColormap(np.vstack([
            mpl.cm.get_cmap('PRGn_r', 10)(np.arange(10))[np.array([4,3,2,1])],
            mpl.cm.get_cmap('bwr', 3)(np.arange(3)),
            mpl.cm.get_cmap('PuOr_r', 10)(np.arange(10))[np.array([8,7,6,5])]]))
        norm0 = Normalize(vmin=1-0.5, vmax=11+0.5)
    if kind==1:
        cmap1 = ListedColormap(mpl.cm.get_cmap('hsv_r', 256)(np.linspace(0., 1., 256))[80:])    
        norm1 = Normalize(vmin=0., vmax=np.pi)    
    
    

    xdim, ydim = sim.dims
    xydata = sim.mxy - sim.center_xy
    points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    vor = freud.locality.Voronoi()
    fpsi = freud.order.Hexatic(k=6, weighted=False)
    if np.size(step)==1:
        ax.set_xlim(-size/2,size/2)
        ax.set_ylim(-size/2,size/2)
        step_txt = sim.times[sample].shape[0] if step[0] == -1 else step[0]
        mxy = xydata[sample, step[0]]
        vor.compute(system=(box, points[sample, step[0]]))
        fpsi.compute(system=(box, points[sample, step[0]]), neighbors=vor.nlist)
        fpsi_phase = np.abs(np.angle(fpsi.particle_order))
        nsides = np.array([polytope.shape[0] for polytope in vor.polytopes])
        
        patches = []
        for polytope in vor.polytopes:
            poly = Polygon(polytope[:,:2], closed=True, facecolor='r')
            patches.append(poly)
        if kind==0:
            collection0 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap0, norm=norm0, alpha=0.6)
            collection0.set_array(nsides)
            ax0 = ax.add_collection(collection0)
            ax.set(title=f'5 and 7-Disclinations (step = {step_txt})')
        if kind==1:
            collection1 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap1, norm=norm1, alpha=0.7)
            collection1.set_array(fpsi_phase)
            ax1 = ax.add_collection(collection1)
            ax.set(title=f'Local Bond-Orientational Order (step = {step_txt})')
        
        ax.scatter(mxy[:,0], mxy[:,1], s=1, c='k', zorder=2)
        box.plot(ax=ax)
        if kind==0:
            cax0 = fig.add_axes([ax.get_position().x1+0.11,ax.get_position().y0+0.045, 
                             0.02, 
                             ax.get_position().height])
                             
            cbar0 = fig.colorbar(ax0, cax=cax0, ticks=np.arange(1, 12))
            cbar0.set_label(label='number of neighbors', labelpad=20., rotation=270)
        if kind==1:
            cax1 = fig.add_axes([ax.get_position().x1+0.11, 
                             ax.get_position().y0-0.02, 
                             0.02, 
                             ax.get_position().height])
            cbar1 = fig.colorbar(ax1, cax=cax1)
            cbar1.set_ticks(np.arange(0., np.pi+(np.pi/4.), np.pi/4.), labels=['0', 'π/4', 'π/2', '3π/4', 'π'])
            cbar1.set_label(label='orientation in radians', labelpad=10., rotation=270)
    
        
   
    else:    
        for i, step_num in enumerate(step):
            ax[i].set_xlim(-size/2,size/2)
            ax[i].set_ylim(-size/2,size/2)
            step_txt = sim.times[sample].shape[0] if step_num == -1 else step_num
            mxy = xydata[sample, step_num]
            vor.compute(system=(box, points[sample, step_num]))
            fpsi.compute(system=(box, points[sample, step_num]), neighbors=vor.nlist)
            fpsi_phase = np.abs(np.angle(fpsi.particle_order))
            nsides = np.array([polytope.shape[0] for polytope in vor.polytopes])
        
            patches = []
            for polytope in vor.polytopes:
                poly = Polygon(polytope[:,:2], closed=True, facecolor='r')
                patches.append(poly)
            if kind==0:
                collection0 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap0, norm=norm0, alpha=0.6)
                collection0.set_array(nsides)
                ax0 = ax[i].add_collection(collection0)
                ax[i].set( title=f'5 and 7-Disclinations (step = {step_txt})')
            if kind==1:
                collection1 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap1, norm=norm1, alpha=0.7)
                collection1.set_array(fpsi_phase)
                ax1 = ax[i].add_collection(collection1)
                ax[i].set(title=f'Local Bond-Orientational Order (step = {step_txt})')
        
            ax[i].scatter(mxy[:,0], mxy[:,1], s=1, c='k', zorder=2)
            box.plot(ax=ax[i])  
        ax[1].set(ylabel=None, yticks=[])
        if kind==0:
            cax0 = fig.add_axes([ax[np.size(step)-1].get_position().x1+0.11, 
                                 ax[np.size(step)-1].get_position().y0+0.045, 
                                 0.02, 
                                 ax[np.size(step)-1].get_position().height])
            cbar0 = fig.colorbar(ax0, cax=cax0, ticks=np.arange(1, 12))
            cbar0.set_label(label='number of neighbors', labelpad=20., rotation=270)
        if kind==1:
            cax1 = fig.add_axes([ax[np.size(step)-1].get_position().x1+0.11, 
                                 ax[np.size(step)-1].get_position().y0-0.02, 
                                 0.02, 
                                 ax[np.size(step)-1].get_position().height])
            cbar1 = fig.colorbar(ax1, cax=cax1)
            cbar1.set_ticks(np.arange(0., np.pi+(np.pi/4.), np.pi/4.), labels=['0', 'π/4', 'π/2', '3π/4', 'π'])
            cbar1.set_label(label='orientation in radians', labelpad=10., rotation=270)
    if savename!=False:
        fig.savefig(savename, format=filetype,bbox_inches='tight')    
    #return ax

def plot_gboop(sim, hloc=None, vline_id=None, yshift=5.0, ylim=(0.,1.), markersize=5, dpi=72,savename=False,filetype='svg'):
    """
    Plots the global bond-orientational order as a function of time/steps.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='w', constrained_layout=True, dpi=dpi)
    
    if isinstance(sim, list):
        for s in sim:
            multiplier = s.multiplier
            timesteps = np.mean(s.times, axis=0) * s.timescale[1]
            xdim, ydim = s.dims
            box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
            sxy = s.mxy - s.center_xy
            points = np.concatenate([sxy, np.zeros((sxy.shape[0], sxy.shape[1], sxy.shape[2], 1))], axis=3)
            fpsi = freud.order.Hexatic(k=6, weighted=False)
            gpsi6 = np.zeros_like(s.times)
            for i, sample in enumerate(points):
                for j, step in enumerate(sample):
                    fpsi.compute(system=(box, step))
                    gpsi6[i,j] = np.mean(np.abs(fpsi.particle_order))
            gpsi6 = np.mean(gpsi6, axis=0)
            
            ax.plot(timesteps, gpsi6, 'o', ms=markersize, c='w', zorder=2)
            ax.plot(timesteps, gpsi6, lw=2., zorder=3)
    else:
        multiplier = sim.multiplier
        timesteps = np.mean(sim.times, axis=0) * sim.timescale[1]
        xdim, ydim = sim.dims
        box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
        xydata = sim.mxy - sim.center_xy
        points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)
        fpsi = freud.order.Hexatic(k=6, weighted=False)
        gpsi6 = np.zeros_like(sim.times)
        for i, sample in enumerate(points):
            for j, step in enumerate(sample):
                fpsi.compute(system=(box, step))
                gpsi6[i,j] = np.mean(np.abs(fpsi.particle_order))
        gpsi6 = np.mean(gpsi6, axis=0)

        ax.plot(timesteps, gpsi6, 'o', ms=markersize, c='w', mec='b', zorder=2)
        ax.plot(timesteps, gpsi6, lw=2., c='b', zorder=3)
    
    if hloc is not None:
        ax.axhline(hloc, ls='--', c='k', lw=1.5, zorder=4)
        ax.annotate(r'$\|\Psi_{{6}}\|(t\rightarrow \infty) \rightarrow ${:.3f}'.format(hloc), 
                    xy=(timesteps[1]*1.1, hloc), xytext=(0., yshift), textcoords='offset pixels', 
                    va='bottom', ha='left', zorder=9)
    else:
        popt = curve_fit(constfunc, timesteps, gpsi6)[0]
        ax.axhline(popt[-1], ls='--', c='k', lw=1.5, zorder=4)
        ax.annotate(r'$\|\Psi_{{6}}\|(t\rightarrow \infty) \rightarrow ${:.3f}'.format(popt[0]), 
                    xy=(timesteps[1]*1.1, constfunc(timesteps[1], *popt)), 
                    xytext=(0., yshift), textcoords='offset pixels', va='bottom', ha='left', zorder=9)
    
    if vline_id is not None:    
        ax.axvline(timesteps[vline_id], lw=1., c='r', zorder=9)
        ax.annotate((f'Thermal equilibrium reached in: \napprox. {timesteps[vline_id]:.5g} s  '
                     f'({(vline_id-1)*multiplier}-{(vline_id+1)*multiplier} steps)'), 
                    xy=(timesteps[vline_id], ylim[1]*0.95), xytext=(-5., 0.), textcoords='offset pixels', 
                    va='top', ha='right', color='r', size=13, zorder=9)
    
    ax.set(xlabel='time $t$', ylabel=r'$\|\Psi_{6}\|$', title=r'Time Evolution of (Magnitude) Global Order Parameter')
    ax.set(xscale='log', xlim=(timesteps[1], timesteps[-1]), ylim=ylim)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()


def plot_bocf(sim, bins=100, step=-1, fit_id=None, dpi=72,savename=False,filetype='svg',start=7,xscale='log',yscale='log',r2fit=1,figsize_=(10,6)):
    """
    Plots the bond-orientational correlation function as a function of distance.
    """
#     sigma = np.sqrt(2. / ((3.**0.5) * sim.rho))
    xdim, ydim = sim.dims
    xydata = sim.mxy - sim.center_xy
    points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)
    
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    fpsi = freud.order.Hexatic(k=6, weighted=False)
    cf = freud.density.CorrelationFunction(bins=bins, r_max=ydim*0.4999)
    cf_vals = []
    for sample in points:
        fpsi.compute(system=(box, sample[step]))
        fpsi_complex = fpsi.particle_order
        cf.compute(system=(box, sample[step]), values=fpsi_complex)    
        cf_edges = cf.bin_edges[:-1] #/ sigma
        cf_vals.append(cf.correlation)
    cf_vals = np.mean(np.abs(np.vstack(cf_vals)), axis=0)
    
    fig, ax = plt.subplots(figsize=figsize_, facecolor='w', constrained_layout=True, dpi=dpi)
    
    ax.plot(cf_edges, cf_vals, 'o', ms=8, c='w', mec='b', zorder=2)
    ax.plot(cf_edges, cf_vals, lw=2.5, c='b', zorder=3) 
    
    cfpks = find_peaks(cf_vals, distance=6)[0]
    idxlim = np.argwhere(cf_edges == cf_edges[cfpks][start])[0][0]-1
    ax.plot(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]], 'x', c='r', ms=14)
    
    fit_func, popt, fit_text = get_cffits(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]], fit_id,r2fit=r2fit,pwrlawconstant=True)
    ax.plot(cf_edges[idxlim:], fit_func(cf_edges[idxlim:], *popt), '--', lw=2., c='k', zorder=9)
    
    ax.set(xlabel=r'$r$', ylabel=f'$g_{{6}}(r)$', title=f'Bond-orientational Correlation Function $g_{{6}}(r)$  |  {fit_text}')
    ax.set(xlim=(cf_edges[cfpks[0]]*0.75, cf_edges[-1]*1.1))
    ax.set(xscale=xscale)
    ax.set(yscale=xscale)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()


def plot_tocf(sim, bins=100, step=-1, fit_id=None, dpi=72,savename=False,filetype='svg',xscale='log',yscale='log',start=7,r2fit=1):     
#     sigma = np.sqrt(2. / ((3.**0.5) * sim.rho))
    xdim, ydim = sim.dims
    xydata = sim.mxy - sim.center_xy
    points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)
    
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    translational_order = freud.order.Translational()
    cf = freud.density.CorrelationFunction(bins=bins, r_max=ydim*0.4999)
    cf_vals = []
    for sample in points:
        translational_order.compute(system=(box, sample[step]))
        torder_complex = translational_order.particle_order  # complex-valued
        cf.compute(system=(box, sample[step]), values=torder_complex)    
        cf_edges = cf.bin_edges[:-1] #/ sigma
        cf_vals.append(cf.correlation)
    cf_vals = np.mean(np.abs(np.vstack(cf_vals)), axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='w', constrained_layout=True, dpi=dpi)
    
    ax.plot(cf_edges, cf_vals, 'o', ms=8, c='w', mec='b', zorder=2)
    ax.plot(cf_edges, cf_vals, lw=2.5, c='b', zorder=3) 
    
    cfpks = find_peaks(cf_vals, distance=6)[0]
    idxlim = np.argwhere(cf_edges == cf_edges[cfpks][start])[0][0]-1
    ax.plot(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]], 'x', c='r', ms=14)
    
    fit_func, popt, fit_text = get_cffits(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]], fit_id,r2fit=r2fit)
    ax.plot(cf_edges[idxlim:], fit_func(cf_edges[idxlim:], *popt), '--', lw=1.5, c='k', zorder=9)
    
    ax.set(xlabel=r'$r$', ylabel=f'$g_{{q}}(r)$', title=f'Translational Correlation Function $g_{{q}}(r)$  |  {fit_text}')
    ax.set(xlim=(cf_edges[cfpks[0]]*0.75, cf_edges[-1]*1.1))
    ax.set(xscale=xscale)
    ax.set(yscale=yscale)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()


def plot_rdf(sim, bins=85, step=-1, peak_distance=1, xlim=None, dpi=72,savename=False,filetype='svg',fit_id=None,showpeak=True,r2fit=True,start=0):
    sigma = np.sqrt(2. / ((3.**0.5) * sim.rho))
    xdim, ydim = sim.dims
    xydata = sim.mxy - sim.center_xy
    points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)
    
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    rdf = freud.density.RDF(bins=bins, r_max=ydim*0.4999)
    rdf_vals = []
    for sample in points:
        rdf.compute(system=(box, sample[step]))
        rdf_edges = rdf.bin_edges[:-1] / sigma
        rdf_vals.append(rdf.rdf)
    rdf_vals = np.mean(np.vstack(rdf_vals), axis=0)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='w', constrained_layout=True, dpi=dpi)
    
    ax.plot(rdf_edges, rdf_vals, 'o', ms=5, c='w', mec='b', zorder=2)
    ax.plot(rdf_edges, rdf_vals, lw=2.5, zorder=3, label=r'$g(r)$')
    
    rdfpks = find_peaks(rdf_vals, distance=peak_distance)[0]
    peaks3 = rdf_edges[rdfpks][:3]
    peakcolors = ['r', 'g', 'b']
    peaknames = [r'$\sigma=$', r'$\sqrt{3}\sigma=$', r'$2\sigma=$']
    peakfactors = [1., np.sqrt(3), 2.]
    for n, (peakcolor, peakname, peakfactor, xpeak) in enumerate(zip(peakcolors, peaknames, peakfactors, peaks3)):
        ax.axvline(xpeak, ls='--', lw=1.5, c=peakcolor, zorder=9, 
                   label=f'Peak {n} @ {xpeak:.5f} ({peakname}{peaks3[0]*peakfactor:.5f})')
    
    if showpeak:
        ax.plot(rdf_edges[rdfpks],rdf_vals[rdfpks],'x', c='r', ms=14)
    fit_func, popt, fit_text = get_cffits(rdf_edges[rdfpks[start:]],rdf_vals[rdfpks[start:]]-1, fit_id,r2fit=r2fit)
    ax.plot(rdf_edges[rdfpks[0]:], fit_func(rdf_edges[rdfpks[0]:], *popt)+1, '--', lw=1.5, c='k', zorder=9)
    if xlim is not None:
        ax.set(xlim=xlim)
    ax.set(xlabel=r'$r / \sigma$', ylabel=f'$g(r)$', title=f'Radial Distribution Function $g(r)$  |  {fit_text}')
    ax.legend(shadow=True)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()


def plot_ssf(sim, bins=50, k_max=8, k_min=0, grid_size=512, step=-1, dpi=72,savename=False,filetype='svg'):
    xdim, ydim = sim.dims
    xydata = sim.mxy - sim.center_xy
    points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)    
    
    box3d = freud.Box.cube(L=ydim) 
    sf = freud.diffraction.StaticStructureFactorDirect(bins=bins, k_max=k_max, k_min=k_min)
    ssf =  freud.diffraction.DiffractionPattern(grid_size=grid_size)
    sf_vals = []
    ssf_vals = []
    for sample in points:
        sf.compute(system=(box3d, sample[step]))
        sf_edges = sf.bin_edges[:-1]
        sf_vals.append(sf.S_k)
        ssf.compute(system=(box3d, sample[step]))
        ssf_vals.append(ssf.to_image())
    sf_vals = np.mean(np.vstack(sf_vals), axis=0)
    ssf_vals = np.mean(np.stack(ssf_vals, axis=0), axis=0).astype('uint8')
    ssfmin, ssfmax = np.min(ssf.k_values), np.max(ssf.k_values)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), facecolor='w', constrained_layout=True, dpi=dpi)
    
    ax[0].plot(sf_edges, sf_vals, 'o', ms=5, c='w', mec='b', zorder=2)
    ax[0].plot(sf_edges, sf_vals, lw=2., zorder=3, label=r'$S(k)$')
    ax[0].set(xlabel=r'$k$', ylabel=r'$S(k)$', title=r'Static Structure Factor')
    
    sfax = ax[1].imshow(ssf_vals, origin='lower', cmap='afmhot', extent=[ssfmin, ssfmax, ssfmin, ssfmax])
    ax[1].set(xlabel=r'$k_{x}$', ylabel=r'$k_{y}$', title=r'2D Diffraction Pattern')
    plt.colorbar(sfax, label=r'$S(\vec{k})$', shrink=0.95, pad=0.02, ax=ax[1])
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()



def plot_realspace(sim, sample=0, step=(0,-1), dpi=72,size=False,savename=False,filetype='svg',dotsize=2):
    """
    Plots 2d realspace images of particles
    """
    if size==False:
        size=sim.dims[0]
    fig, ax = plt.subplots(1,2, figsize=(10, 5), facecolor='w', constrained_layout=True, dpi=dpi)
    
    xdim, ydim = sim.dims
    xydata = sim.mxy - sim.center_xy
    points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)
    for i, step_num in enumerate(step):
        mxy = xydata[sample, step_num]
        ax[i].scatter(mxy[:,0], mxy[:,1], s=dotsize, c='k', zorder=2)
        
        step_txt = sim.times[sample].shape[0] if step_num == -1 else step_num
        ax[i].set(title=f'Real Space (step = {step_txt})')
        ax[i].set_xlim(-size/2,size/2)
        ax[i].set_ylim(-size/2,size/2)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    
def animate_realspace(sim,sample=0,dpi=72,size=False,frames=300):
    if size==False:
        size=sim.dims[0]
    fig,ax=plt.subplots()
    line,=ax.plot([],[],marker = ".",markersize = 1,linestyle='none',color='black')
    xydata = sim.mxy - sim.center_xy
    ax.set_xlim(-size/2,size/2)
    ax.set_ylim(-size/2,size/2)
    def animate(frame_number):
        mxy = xydata[0, frame_number]
        x = mxy[:,0]
        y = mxy[:,1]
        line.set_data(x, y)
        ax.set_title(frame_number)
        return line
    anim=FuncAnimation(fig,animate,frames=frames,interval=100)
    display(HTML(anim.to_jshtml()))
def plot_ssf2(sim, bins=50, k_max=8, k_min=0, grid_size=512, step=-1, dpi=72,lower=0,upper=1,savename=False,filetype='svg',sample=0):
    xdim, ydim = sim.dims
    xydata = sim.mxy - sim.center_xy
    points = np.concatenate([xydata, np.zeros((xydata.shape[0], xydata.shape[1], xydata.shape[2], 1))], axis=3)    
    
    box3d = freud.Box.cube(L=ydim) 
    ssf =  freud.diffraction.DiffractionPattern(grid_size=grid_size)
    ssf_vals = []
    sample=points[sample]
    ssf.compute(system=(box3d, sample[step]))
    ssf_vals.append(ssf.to_image(vmin=lower*ssf.N_points,vmax=upper*ssf.N_points))
    ssf_vals = np.mean(np.stack(ssf_vals, axis=0), axis=0).astype('uint8')
    ssfmin, ssfmax = np.min(ssf.k_values), np.max(ssf.k_values)
    
    fig, ax = plt.subplots(1, 1, facecolor='w', constrained_layout=True, dpi=dpi)
    sfax = ax.imshow(ssf_vals, origin='lower', cmap='afmhot', extent=[ssfmin, ssfmax, ssfmin, ssfmax])
    ax.set(xlabel=r'$k_{x}$', ylabel=r'$k_{y}$', title=r'2D Diffraction Pattern'+f' Step {step}')
    plt.colorbar(sfax, label=r'$S(\vec{k})$', shrink=0.95, pad=0.02, ax=ax)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()