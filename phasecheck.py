#-----------------------------------------------------------------------------80
# NOTE: DOCUMENTATION FOR THIS FILE IS STILL IN PROGRESS

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
from PIL import Image
import trackpy as tp
from sxmreader import SXMReader



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

def get_cffits(x, y, fit_id,sigma=None,r2fit=1,pwrlawconstant=False):
    popt_constfunc = curve_fit(constfunc, x, y,sigma=sigma,maxfev=3000, bounds=(-np.amin(x), np.inf))[0]
    if pwrlawconstant:
        pwrlaw=pwrlawdecay2
        popt_pwrlawdecay = curve_fit(pwrlaw, x, y,sigma=sigma,
                                     maxfev=3000,p0=(10,0,1),bounds=(-np.amin(x),np.inf))[0]
   
    else:
        pwrlaw=pwrlawdecay
        popt_pwrlawdecay = curve_fit(pwrlaw, x, y,sigma=sigma,maxfev=3000,p0=(10,1),
                                     bounds=(-np.amin(x),np.inf))[0]
   


    popt_expdecay = curve_fit(expdecay, x, y,sigma=sigma,
                              maxfev=3000,p0=(np.amax(x),1), bounds=(0, np.inf))[0]
    
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
        r'$r^{{-\eta}}: $ ($\eta\simeq ${:.4f})'.format(popts[1][-1]),
        r'$\exp(-r/\xi): $($\xi\simeq ${:.4f})'.format(popts[2][-1])
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

class imgdata:
    def __init__(self,path,filetype='SXM',**kwargs,):
        if filetype=='SXM':
            img = SXMReader(path, correct='lines')[0]
            self.sxm=True
            self.dims=[
                SXMReader(path).scans[0].size['real']['x']*10**9,
                SXMReader(path).scans[0].size['real']['y']*10**9]

        else:
            img = Image.open(path)
            img = img.resize((200,200), Image.ANTIALIAS)
            img=img.convert('L')
            img.shape=img.size
            img.ndim=2
            self.sxm=False
        self.img=img
        self.data=tp.locate(img,**kwargs).reset_index()
        
    
def plot_boops(img_d,dpi=72,savename=False,kind=0,realdim=False,filetype='svg',anim_param=False):
    """
    Plots the local bond-orientational order parameter to show 5-7 disclinations and
    orientational order in radians.
    """
    
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]

     
    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])

    if not anim_param:
        fig, ax = plt.subplots(1,figsize=(7,6), facecolor='w', constrained_layout=True, dpi=dpi)
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
    
   
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    vor = freud.locality.Voronoi()
    fpsi = freud.order.Hexatic(k=6, weighted=False)
    ax.set_xlim(0,xdim)
    ax.set_ylim(0,ydim)
    if True:
        mxy = xydata
        vor.compute(system=(box, points))
        fpsi.compute(system=(box, points), neighbors=vor.nlist)
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
            ax.set_title(label=f'5 and 7-Disclinations ',fontsize=20)
        if kind==1:
            collection1 = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap1, norm=norm1, alpha=0.7)
            collection1.set_array(fpsi_phase)
            ax1 = ax.add_collection(collection1)
            ax.set_title(label=f'Local Bond-Orientational Order ',fontsize=20)
        
        ax.scatter(mxy[:,0], mxy[:,1], s=1, c='k', zorder=2)
        #box.plot(ax=ax)
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
            cbar1.set_ticks(np.arange(0., np.pi+(np.pi/4.), np.pi/4.))
            cbar1.ax.set_yticklabels(labels=['0', 'π/4', 'π/2', '3π/4', 'π'])
            cbar1.set_label(label='orientation in radians', labelpad=10., rotation=270)
    
    if savename!=False:
        fig.savefig(savename, format=filetype,bbox_inches='tight')    
    #return ax

def plot_bocf(img_d, bins=100, fit_id=None,dist=6, dpi=72,savename=False,filetype='svg',realdim=False,start=7,xtype='linear',ytype='linear',r2fit=1,figsize_=(10,6)):
    """
    Plots the bond-orientational correlation function as a function of distance.
    """
#     sigma = np.sqrt(2. / ((3.**0.5) * sim.rho))
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]

     
    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])

    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    fpsi = freud.order.Hexatic(k=6, weighted=False)
    cf = freud.density.CorrelationFunction(bins=bins, r_max=ydim*0.4999)
    fpsi.compute(system=(box, points))
    fpsi_complex = fpsi.particle_order
    cf.compute(system=(box, points), values=fpsi_complex)    
    cf_edges = cf.bin_edges[:-1] #/ sigma
    cf_vals=cf.correlation
    
    fig, ax = plt.subplots(figsize=figsize_, facecolor='w', constrained_layout=True, dpi=dpi)
    
    ax.plot(cf_edges, cf_vals, 'o', ms=8, c='w', mec='b', zorder=2)
    ax.plot(cf_edges, cf_vals, lw=2.5, c='b', zorder=3) 
    
    cfpks = find_peaks(cf_vals, distance=dist)[0]
    idxlim = np.argwhere(cf_edges == cf_edges[cfpks][start])[0][0]-1
    ax.plot(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]], 'x', c='r', ms=14)
    
    fit_func, popt, fit_text = get_cffits(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]], fit_id,r2fit=r2fit,pwrlawconstant=True)
    ax.plot(cf_edges[idxlim:], fit_func(cf_edges[idxlim:], *popt), '--', lw=2., c='k', zorder=9)
    
    ax.set(xlabel=r'$r$', ylabel=f'$g_{{6}}(r)$', title=f'Bond-orientational Correlation Function $g_{{6}}(r)$  |  {fit_text}')
    ax.set_title(
        label=f'Bond-orientational Correlation Function $g_{{6}}(r)$  |  {fit_text}',
        fontsize=20)
    ax.set(xlim=(cf_edges[cfpks[0]]*0.75, cf_edges[-1]*1.1))
    ax.set(xscale=xtype)
    ax.set(yscale=xtype)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()


def plot_tocf(img_d, bins=100,realdim=False, fit_id=None, dist=6,dpi=72,savename=False,filetype='svg',xtype='linear',ytype='linear',start=1,r2fit=1):     
#     sigma = np.sqrt(2. / ((3.**0.5) * sim.rho))
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]

     
    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])

    
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    translational_order = freud.order.Translational()
    cf = freud.density.CorrelationFunction(bins=bins, r_max=ydim*0.4999)
    cf_vals = []
    translational_order.compute(system=(box, points))
    torder_complex = translational_order.particle_order  # complex-valued
    cf.compute(system=(box, points), values=torder_complex)    
    cf_edges = cf.bin_edges[:-1] #/ sigma
    cf_vals=cf.correlation
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='w', constrained_layout=True, dpi=dpi)
    
    ax.plot(cf_edges, cf_vals, 'o', ms=8, c='w', mec='b', zorder=2)
    ax.plot(cf_edges, cf_vals, lw=2.5, c='b', zorder=3) 
    
    cfpks = find_peaks(cf_vals, distance=dist)[0]
    idxlim = np.argwhere(cf_edges == cf_edges[cfpks][start])[0][0]-1
    ax.plot(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]], 'x', c='r', ms=14)
    
    fit_func, popt, fit_text = get_cffits(cf_edges[cfpks[start:]], cf_vals[cfpks[start:]], fit_id,r2fit=r2fit)
    ax.plot(cf_edges[idxlim:], fit_func(cf_edges[idxlim:], *popt), '--', lw=1.5, c='k', zorder=9)
    
    ax.set(xlabel=r'$r$', ylabel=f'$g_{{q}}(r)$', title=f'Translational Correlation Function $g_{{q}}(r)$  |  {fit_text}')
    ax.set_title(label=f'Translational Correlation Function $g_{{q}}(r)$  |  {fit_text}',
                fontsize=20)
    ax.set(xlim=(cf_edges[cfpks[0]]*0.75, cf_edges[-1]*1.1))
    ax.set(xscale=xtype)
    ax.set(yscale=xtype)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()


def plot_rdf(img_d, bins=85, realdim=False, step=-1,rho=1, peak_distance=1, xlim=None, dpi=72,savename=False,filetype='svg',fit_id=None,showpeak=True,r2fit=True,start=0):
    sigma = np.sqrt(2. / ((3.**0.5) * rho))
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]

     
    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])

    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    rdf = freud.density.RDF(bins=bins, r_max=ydim*0.4999)

    rdf.compute(system=(box, points))
    rdf_edges = rdf.bin_edges[:-1] / sigma
    rdf_vals=rdf.rdf

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


def plot_ssf(img_d, bins=50, k_max=8, k_min=0, grid_size=512,dpi=72,realdim=False,savename=False,filetype='svg'):
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]

     
    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])


    box3d = freud.Box.cube(L=ydim) 
    sf = freud.diffraction.StaticStructureFactorDirect(bins=bins, k_max=k_max, k_min=k_min)
    ssf =  freud.diffraction.DiffractionPattern(grid_size=grid_size)
    sf_vals = []
    ssf_vals = []
    sf.compute(system=(box3d, points))
    sf_edges = sf.bin_edges[:-1]
    sf_vals=sf.S_k
    ssf.compute(system=(box3d, points))
    ssf_vals=ssf.to_image()
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



def plot_ssf2(img_d, bins=50, k_max=8, k_min=0, grid_size=512,  dpi=72,lower=0.01,upper=1,realdim=False,savename=False,filetype='svg',sample=0):
    if realdim==False:
        xdim, ydim = img_d.img.shape
        xscale=1
        yscale=1
    else:
        if img_d.sxm==True:
            xdim=img_d.dims[0]
            ydim=img_d.dims[1]
        else:
            xdim=realdim[0]
            ydim=realdim[1]
        xscale=xdim/img_d.img.shape[0]
        yscale=ydim/img_d.img.shape[1]

     
    xydata=np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale] for i in range(img_d.data['x'].size)  ])
    points =np.array([[img_d.data['x'][i]*xscale,img_d.data['y'][i]*yscale,0] for i in range(img_d.data['x'].size)  ])

    box3d = freud.Box.cube(L=ydim) 
    ssf =  freud.diffraction.DiffractionPattern(grid_size=grid_size)
    ssf_vals = []
    sample=points
    ssf.compute(system=(box3d, sample))
    ssf_vals=ssf.to_image(vmin=lower*ssf.N_points,vmax=upper*ssf.N_points)
    ssfmin, ssfmax = np.min(ssf.k_values), np.max(ssf.k_values)
    
    fig, ax = plt.subplots(1, 1, facecolor='w', constrained_layout=True, dpi=dpi)
    sfax = ax.imshow(ssf_vals, origin='lower', cmap='afmhot', extent=[ssfmin, ssfmax, ssfmin, ssfmax])
    ax.set(xlabel=r'$k_{x}$', ylabel=r'$k_{y}$', title=r'2D Diffraction Pattern')
    plt.colorbar(sfax, label=r'$S(\vec{k})$', shrink=0.95, pad=0.02, ax=ax)
    if savename!=False:
        fig.savefig(savename, format=filetype)
    return plt.show()