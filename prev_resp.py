import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d, circmean
# from load import load_online
import uniformised_space_utils as usu
import pickle
from tqdm import tqdm

def angles_from_counts(counts, bins):
  """ returns list of angles corresponding to the given number of ''counts'' for each angle """
  angles_radians = np.linspace(-np.pi/2, np.pi/2, bins, endpoint=True)  # angles in radians
  return [[j for a in [[angles_radians[i]]*int(counts[i,k]) for i in range(bins)] for j in a] for k in range(bins)]

def circmean_2(angles,space=3,o_adjust=2):
    langle = len(angles)
    cosm = np.zeros(langle)
    sinm = np.zeros(langle)

    for ang in range(langle):
        angle = np.array([r for a in angles[ang] for r in a])/(360.0/o_adjust)*np.pi*2
        c = np.cos(angle)
        s = np.sin(angle)
        cosm[ang] = np.mean(c)
        sinm[ang] = np.mean(s)
    argmean = np.array([np.arctan2(np.average(sinm.take(range(i-space,i+space),mode='wrap')),np.average(cosm.take(range(i-space,i+space),mode='wrap'))) for i in range(langle)])
    return argmean / np.pi *180.0 /o_adjust

def serial_dep(gg):
    sdep = np.array([np.deg2rad(gg['allTarget'][:-1])-np.deg2rad(gg['allTarget'][1:]),
                 np.deg2rad(gg['allResponse'][1:])-np.deg2rad(gg['allTarget'][1:])])

    sdep[0] = np.where(sdep[0] < -np.pi/2, sdep[0]+np.pi, sdep[0])
    sdep[0] = np.where(sdep[0] > np.pi/2, sdep[0]-np.pi, sdep[0])
    sdep[1] = np.where(sdep[1] < -np.pi/2, sdep[1]+np.pi, sdep[1])
    sdep[1] = np.where(sdep[1] > np.pi/2, sdep[1]-np.pi, sdep[1])

    delshort = [sdep[0][np.argwhere(gg['allDelay'][1:] < 1)], sdep[1][np.argwhere(gg['allDelay'][1:] < 1)]]
    dellong = [sdep[0][np.argwhere(gg['allDelay'][1:] > 4)], sdep[1][np.argwhere(gg['allDelay'][1:] > 4)]]

    allresp = np.where(np.deg2rad(gg['allResponse']) > np.pi/2, np.deg2rad(gg['allResponse'])-np.pi, np.deg2rad(gg['allResponse']))
    allresp = np.where(allresp < -np.pi/2, allresp+np.pi, allresp)
    allstim = np.where(np.deg2rad(gg['allTarget']) > np.pi/2, np.deg2rad(gg['allTarget'])-np.pi, np.deg2rad(gg['allTarget']))
    allstim = np.where(allstim < -np.pi/2, allstim+np.pi, allstim)
    sdep[1] = np.array(sdep[1])

    binning = 45

    histo = plt.hist2d(sdep[0],sdep[1],bins=binning,cmap='plasma')
    histoct = angles_from_counts(histo[0], bins=binning)
    print(len(histoct))
    plt.xlabel('relative orientation of previous response ($R_{n-1}-S_n$)')
    plt.plot(np.linspace(-np.pi/2,np.pi/2,binning),[circmean(histoct[i][:],high=np.pi,low=-np.pi) for i in range(binning)],'c--')
    histallrow = np.empty(binning,dtype=list)
    angspace = np.linspace(-np.pi/2,np.pi/2,binning)
    histagain = histo[0]
    for m in range(binning):
        histallrow[m] = [int(histagain[m][k]*10e2)*[angspace[k]] for k in range(binning)]
    hrowavg = circmean_2(histallrow,space=1, o_adjust=2)
    plt.hlines(0,-np.pi/2,np.pi/2,'k')
    plt.ylabel('error on current trial ($R_n-S_n$)')
    plt.plot(angspace, hrowavg, 'w--')
    plt.colorbar()
    plt.ylim(-1,1)
    plt.show()
    #plt.plot(np.linspace(-np.pi/2,np.pi/2,50),np.deg2rad(circrowquick(histo[0],correct=2,binsize=50)),'c--')
    return allstim, allresp

def response_sampling(allresp, binning=31):
    # usual density histogram
    fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[4,1]})
    bins = ax[1,0].hist2d(allresp[1:], allresp[:-1], bins=binning, cmap ='plasma')
    ax[1,0].set_xlabel('$R_{n-1}$')
    ax[1,0].set_ylabel('$R_{n}$')
    ax[1,1].margins(y=0)
    ax[1,1].set_yticks([])
    ax[1,1].axis('off')
    ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=0),width=np.pi/(binning-1))
    ax[0,0].margins(x=0)
    ax[0,0].grid(False)
    ax[1,1].grid(False)
    ax[0,0].axis('off')
    ax[0,0].set_xticks([])
    ax[0,1].set_visible(False)
    ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=1),height=np.pi/(binning-1))
    fig.colorbar(bins[3],ax=ax[1,1])
    plt.show()

    print(np.sum(bins[0][:15,:15]), np.sum(bins[0][:15,-15:]), np.sum(bins[0][-15:,15:]), np.sum(bins[0][-15:,-15:]))
    # need marginals!!

    bins = plt.hist2d(allstim[1:], allresp[:-1], bins=binning, cmap ='plasma')
    plt.xlabel('$S_n$')
    plt.ylabel('$R_{n-1}$')
    plt.colorbar()
    plt.show()


    bins = plt.hist2d(allstim[:-1], allresp[1:], bins=binning, cmap ='plasma')
    plt.xlabel('$S_{n-1}$')
    plt.ylabel('$R_{n}$')
    plt.colorbar()
    plt.show()

    # marginal rearrangement
    bins = plt.hist2d(allresp, np.random.permutation(allresp), bins=binning, cmap ='plasma')
    plt.xlabel('$R_n$')
    plt.ylabel('$R_{n-1}$')
    plt.colorbar()
    plt.show()

    bins = plt.hist2d(np.random.choice(allresp, size = len(allresp), replace='True')[:-1], allresp[1:], bins=binning, cmap ='plasma')
    plt.ylabel('$R_n$')
    plt.xlabel('$R_{resampled}$')
    plt.colorbar()
    plt.show()

    fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[4,1]})
    bins = ax[1,0].hist2d(np.random.choice(allresp, size = len(allresp), replace='True')[:-1], allresp[1:], bins=binning, cmap ='plasma')
    ax[1,0].set_ylabel('$R_resampled$')
    ax[1,0].set_xlabel('$R_{n}$')
    ax[1,1].margins(y=0)
    ax[1,1].set_yticks([])
    ax[1,1].axis('off')
    ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=0),width=np.pi/(binning-1))
    ax[0,0].margins(x=0)
    ax[0,0].grid(False)
    ax[1,1].grid(False)
    ax[0,0].axis('off')
    ax[0,0].set_xticks([])
    ax[0,1].set_visible(False)
    ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=1),height=np.pi/(binning-1))
    fig.colorbar(bins[3],ax=ax[1,1])
    plt.show()
    return

def circmean_default(vals, high=np.pi/2, low=-np.pi/2):
    return circmean(vals, high=high, low=low)

def histo_fxn(allstim, allresp, binning=31):
    # histogram coloured by rn-sn...
    diff1 = allresp[1:]-allstim[1:]
    diff1 = np.where(diff1 > np.pi/2, diff1-np.pi, diff1)
    diff1 = np.where(diff1 < -np.pi/2, diff1+np.pi, diff1)

    diff = diff1

    sign_quad = np.sign(allstim[1:]) - np.sign(allresp[:-1])

    ang_sep = np.where(sign_quad == 0, allstim[1:]-allresp[:-1], allstim[1:]+allresp[:-1])

    H, xedges, yedges, binnumber = binned_statistic_2d(allstim[1:], ang_sep, values=diff, 
                                                    statistic=circmean_default, bins=binning,
                                    expand_binnumbers=True)
    H = np.ma.masked_where(H==0, H)
    XX, YY = np.meshgrid(xedges, yedges)

    plt.pcolormesh(XX,YY,H.T,cmap='RdBu',vmin=-0.4,vmax=0.4)
    plt.colorbar()
    plt.xlabel('$S_{n}$')
    plt.ylabel('$S_{n}-R_{n-1}$ ($R_{n-1}$ reflected to same quadrant as $S_n$)')
    #plt.ylabel('$R_{n-1}$')
    plt.title('$\mathbb{E}[(R_{n-1} - R_n)_w + (R_{n}-S_n)_w]$')
    plt.title('$\mathbb{E}[R_{n}-S_{n}]$')
    plt.show()
    return ang_sep, diff

def lookup_hist_value(x_val, y_val, x_edges=20, y_edges=20, hist_matrix=np.zeros((20, 20))):
    # Find the bin index for x and y
    x_idx = np.searchsorted(x_edges, x_val, side='right') - 1
    y_idx = np.searchsorted(y_edges, y_val, side='right') - 1

    # Check for out-of-bounds
    if 0 <= x_idx < hist_matrix.shape[1] and 0 <= y_idx < hist_matrix.shape[0]:
        return hist_matrix[y_idx, x_idx]
    else:
        if x_idx > len(x_edges)-2:
            x_idx = 0
        
        elif y_idx > len(y_edges)-2:
            y_idx = 0
        
        elif x_idx < 0:
            x_idx = len(x_edges) - 2
        
        elif y_idx < 0:
            y_idx = len(y_edges) - 2
        return hist_matrix[y_idx, x_idx] 
    

def circvar(angles,space=3, o_adjust=np.pi/180):
    langle = len(angles)
    cosm = np.zeros(langle)
    sinm = np.zeros(langle)
    delt = np.zeros(langle)
    space = 4

    for ang in range(langle):
        angle = np.array([r for a in angles[ang] for r in a])/(360.0/o_adjust)*np.pi*2
        c = np.cos(angle)
        s = np.sin(angle)
        cosm[ang] = np.mean(c)
        sinm[ang] = np.mean(s)
        r = np.sqrt(cosm[ang]**2+sinm[ang]**2)
        delt[ang] = 1-r#(1-np.mean(np.cos(2*(angle-argm))))/2*r
    argvar = np.array([np.average(delt.take(range(i-space,i+space),mode='wrap')) for i in range(langle)])
    return argvar  / np.pi *180.0 /o_adjust

def model_plot(samps, resps, binsize=41, plot_lines=True, filename='none'):
    fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[3,1]})
    histo = ax[1,0].hist2d(samps, resps, bins = binsize, cmap='plasma', density=True, vmax=1.3,vmin=0)
    ax[1,1].barh(y=np.linspace(0,180,binsize),width=np.sum(histo[0],axis=0),height=185/binsize)
    fig.colorbar(histo[3],ax=ax[1,1])
    ax[1,1].margins(y=0)
    ax[1,1].set_yticks([])
    ax[1,1].axis('off')
    ax[0,0].bar(x=np.linspace(0,180,binsize),height=np.sum(histo[0],axis=1),width=185/binsize)
    ax[0,0].margins(x=0)
    ax[0,0].grid(False)
    ax[1,1].grid(False)
    ax[0,0].axis('off')
    ax[0,0].set_xticks([])
    ax[0,1].set_visible(False)
    # ax[1,0].set_title(title,fontsize=20,style='oblique')
    ax[1,0].set_xlabel('stimulus S',fontsize=12)
    ax[1,0].set_ylabel('response R',fontsize=12)
    if plot_lines == True:
        histagain = histo[0]
        angspace = np.linspace(-np.pi/2,np.pi/2,binsize)
        histallrow = np.empty(binsize,dtype=list)
        for m in (range(binsize)):
            histallrow[m] = [int(histagain[m][k]*10e5)*[angspace[k]] for k in range(binsize)]
        hrowavg = circmean_2(histallrow,o_adjust=360/np.pi,space=1)
        histall = np.empty(binsize,dtype=list)
        for m in (range(binsize)):
            histall[m] = [int(histagain[k][m]*10e5)*[angspace[k]] for k in range(binsize)]
        hcolavg = circmean_2(histall,o_adjust=360/np.pi,space=1)
        hrowavg = np.where(hrowavg < -np.pi/2, hrowavg + np.pi, hrowavg)
        hrowavg = np.where(hrowavg > np.pi/2, hrowavg - np.pi, hrowavg)
        hrowavg[0] = np.where(hrowavg[0] > 0, hrowavg[0]-np.pi, hrowavg[0])
        hcolavg = np.where(hcolavg < -np.pi/2, hcolavg + np.pi, hcolavg)
        hcolavg = np.where(hcolavg > np.pi/2, hcolavg - np.pi, hcolavg)
        hcolavg[0] = np.where(hcolavg[0] > 0, hcolavg[0]-np.pi, hcolavg[0])
        hvar = circvar(histall,space=1,o_adjust=360/np.pi)
        hvar_rev = circvar(histallrow,space=1,o_adjust=360/np.pi)
        ax[1,0].plot(angspace,angspace, color= 'gray')
        # ax[1,0].plot(hcolavg,angspace,color='cyan',linestyle='--',label='row mean')
        ax[1,0].plot(angspace,hrowavg,color='white',linestyle='--', label='bias')
        ax[1,0].legend()
        print('fig gen done')
    plt.savefig(filename+'.png', dpi=150)
    plt.show()

    plt.plot(angspace,hcolavg-angspace,label='mean [decision] (rows)')
    plt.plot(angspace,hrowavg-angspace,label='mean [bias] (columns)')
    plt.ylabel('error (radians, 0.1 rad is approx 5.7 deg)')
    plt.xlabel('angle')
    plt.ylim(-0.25, 0.25)
    plt.legend()
    plt.savefig(filename+'_bias.png', dpi=150)
    plt.show()

    plt.plot(angspace,hvar,label='variance (rows)')
    plt.plot(angspace,hvar_rev,label='variance (columns)')
    plt.ylabel('error (radians, 0.1 rad is approx 5.7 deg)')
    plt.xlabel('angle')
    plt.ylim(0, 0.2)
    plt.legend()
    plt.savefig(filename+'_var.png', dpi=150)
    plt.show()
    return


def allComparisonHists(allstim,allresp,other=None,label=['$S_{n}$','$R_{n-1}$','$R_n$'], filename='none'):

    # histogram coloured by rn-sn...
        
    binning = 31
    diff1 = allresp[1:]-allstim[1:]
    diff1 = np.where(diff1 > np.pi/2, diff1-np.pi, diff1)
    diff1 = np.where(diff1 < -np.pi/2, diff1+np.pi, diff1)

    diff = diff1

    sign_quad = np.sign(allstim[1:]) - np.sign(other)

    ang_sep = np.where(sign_quad == 0, allstim[1:]-other, allstim[1:]+other)

    #ang_sep = np.abs(allstim[1:]) - np.abs(allresp[:-1])
    def circmean_default(vals, high=np.pi/2, low=-np.pi/2):
        return circmean(vals, high=high, low=low)

    H, xedges, yedges, binnumber = binned_statistic_2d(allstim[1:], ang_sep, values=diff, 
                                                    statistic=circmean_default, bins=binning,
                                    expand_binnumbers=True)
    H = np.ma.masked_where(H==0, H)
    XX, YY = np.meshgrid(xedges, yedges)

    plt.pcolormesh(XX,YY,H.T,cmap='RdBu',vmin=-0.4,vmax=0.4)
    plt.colorbar()
    plt.xlabel(label[0])
    plt.ylabel(label[0]+'-'+label[1]+'('+label[1]+' reflected to same quadrant as '+label[0]+')')
    #plt.ylabel('$R_{n-1}$')
    plt.title('$\mathbb{E}[(R_{n-1} - R_n)_w + (R_{n}-S_n)_w]$')
    plt.title('$\mathbb{E}[R_{n}-S_{n}]$')
    plt.savefig(filename+'_attract.png', dpi=150)
    plt.show()

    binning = 31
    diff1 = allresp[1:]-allstim[1:]
    diff1 = np.where(diff1 > np.pi/2, diff1-np.pi, diff1)
    diff1 = np.where(diff1 < -np.pi/2, diff1+np.pi, diff1)

    diff = diff1

    sign_quad = np.sign(allstim[1:]) - np.sign(other)

    ang_sep = np.where(sign_quad == 0, allstim[1:]-other, allstim[1:]-other)
    ang_sep = np.where(ang_sep > np.pi/2, ang_sep-np.pi, ang_sep)
    ang_sep = np.where(ang_sep < -np.pi/2, ang_sep+np.pi, ang_sep)

    #ang_sep = np.abs(allstim[1:]) - np.abs(allresp[:-1])
    def circmean_default(vals, high=np.pi/2, low=-np.pi/2):
        return circmean(vals, high=high, low=low)

    H, xedges, yedges, binnumber = binned_statistic_2d(allstim[1:], ang_sep, values=diff, 
                                                    statistic=circmean_default, bins=binning,
                                    expand_binnumbers=True)
    H = np.ma.masked_where(H==0, H)
    XX, YY = np.meshgrid(xedges, yedges)

    plt.pcolormesh(XX,YY,H.T,cmap='RdBu',vmin=-0.4,vmax=0.4)
    plt.colorbar()
    plt.xlabel(label[0])
    plt.ylabel(label[0]+'-'+label[1])
    #plt.ylabel('$R_{n-1}$')
    plt.title('$\mathbb{E}[(R_{n-1} - R_n)_w + (R_{n}-S_n)_w]$')
    plt.title('$\mathbb{E}[R_{n}-S_{n}]$')
    plt.savefig(filename+'_attract_no_reflect.png', dpi=150)
    plt.show()

    binning = 31
    # usual density histogram
    fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[4,1]})
    bins = ax[1,0].hist2d(other, allresp[1:], bins=binning, cmap ='plasma', density=True, vmax=0.3, vmin=0)
    ax[1,0].set_xlabel(label[1])
    ax[1,0].set_ylabel(label[2])
    ax[1,1].margins(y=0)
    ax[1,1].set_yticks([])
    ax[1,1].axis('off')
    ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=1),width=np.pi/(binning-1))
    ax[0,0].margins(x=0)
    ax[0,0].grid(False)
    ax[1,1].grid(False)
    ax[0,0].axis('off')
    ax[0,0].set_xticks([])
    ax[0,1].set_visible(False)
    ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=0),height=np.pi/(binning-1))
    fig.colorbar(bins[3],ax=ax[1,1])
    plt.savefig(filename+'_rnrn.png', dpi=150)
    plt.show()

    fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[4,1]})
    bins = ax[1,0].hist2d( other, np.random.choice(allresp, size = len(allresp), replace='True')[:-1], bins=binning, 
                          cmap ='plasma', density=True, vmax=0.3, vmin=0)
    ax[1,0].set_ylabel(label[2]+' resampled')
    ax[1,0].set_xlabel(label[1])
    ax[1,1].margins(y=0)
    ax[1,1].set_yticks([])
    ax[1,1].axis('off')
    ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=1),width=np.pi/(binning-1))
    ax[0,0].margins(x=0)
    ax[0,0].grid(False)
    ax[1,1].grid(False)
    ax[0,0].axis('off')
    ax[0,0].set_xticks([])
    ax[0,1].set_visible(False)
    ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=0),height=np.pi/(binning-1))
    fig.colorbar(bins[3],ax=ax[1,1])
    plt.savefig(filename+'_rnrn_resamp.png', dpi=150)
    plt.show()

    return


def allComparisonHists_norm(allstim,allresp,other=None,label=['$S_{n}$','$R_{n-1}$','$R_n$']):

    # histogram coloured by rn-sn...
        
    binning = 31
    diff1 = allresp[1:]-allstim[1:]
    diff1 = np.where(diff1 > np.pi/2, diff1-np.pi, diff1)
    diff1 = np.where(diff1 < -np.pi/2, diff1+np.pi, diff1)

    diff = diff1

    sign_quad = np.sign(allstim[1:]) - np.sign(other)

    ang_sep = np.where(sign_quad == 0, allstim[1:]-other, allstim[1:]+other)

    #ang_sep = np.abs(allstim[1:]) - np.abs(allresp[:-1])
    def circmean_default(vals, high=np.pi/2, low=-np.pi/2):
        return circmean(vals, high=high, low=low)

    H, xedges, yedges, binnumber = binned_statistic_2d(allstim[1:], ang_sep, values=diff, 
                                                    statistic=circmean_default, bins=binning,
                                    expand_binnumbers=True)
    H = np.ma.masked_where(H==0, H)
    XX, YY = np.meshgrid(xedges, yedges)

    plt.pcolormesh(XX,YY,H.T,cmap='RdBu',vmin=-0.4,vmax=0.4)
    plt.colorbar()
    plt.xlabel(label[0])
    plt.ylabel(label[0]+'-'+label[1]+'('+label[1]+' reflected to same quadrant as '+label[0]+')')
    #plt.ylabel('$R_{n-1}$')
    plt.title('$\mathbb{E}[(R_{n-1} - R_n)_w + (R_{n}-S_n)_w]$')
    plt.title('$\mathbb{E}[R_{n}-S_{n}]$')
    plt.show()

    binning = 31
    # usual density histogram
    fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[4,1]})
    bins = np.histogram2d(other, allresp[1:], bins=binning, density=True)
    newbins = bins[0]/np.sum(bins[0],axis=1)
    ax[1,0].imshow(newbins,cmap ='plasma', origin='lower')
    ax[1,0].set_xlabel(label[1])
    ax[1,0].set_ylabel(label[2])
    ax[1,1].margins(y=0)
    ax[1,1].set_yticks([])
    ax[1,1].axis('off')
    ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(newbins.T,axis=1),width=np.pi/(binning-1))
    ax[0,0].margins(x=0)
    ax[0,0].grid(False)
    ax[1,1].grid(False)
    ax[0,0].axis('off')
    ax[0,0].set_xticks([])
    ax[0,1].set_visible(False)
    ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(newbins.T,axis=0),height=np.pi/(binning-1))
    # fig.colorbar(np.linspace(0,1),ax=ax[1,1])
    plt.show()

    fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[4,1]})
    bins = np.histogram2d(other, np.random.choice(allresp, size = len(allresp), replace='True')[:-1], bins=binning, density=True)
    newbins = bins[0]/np.sum(bins[0],axis=1)
    ax[1,0].imshow(newbins,cmap ='plasma', origin='lower')
    ax[1,0].set_ylabel(label[2]+' resampled')
    ax[1,0].set_xlabel(label[1])
    ax[1,1].margins(y=0)
    ax[1,1].set_yticks([])
    ax[1,1].axis('off')
    ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(newbins.T,axis=1),width=np.pi/(binning-1))
    ax[0,0].margins(x=0)
    ax[0,0].grid(False)
    ax[1,1].grid(False)
    ax[0,0].axis('off')
    ax[0,0].set_xticks([])
    ax[0,1].set_visible(False)
    ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(newbins.T,axis=0),height=np.pi/(binning-1))
    # fig.colorbar(bins[3],ax=ax[1,1])
    plt.show()

    return


if __name__ == '__main__':
    gg, ggpart = load_online(name='peter',twowm=True,delay=False)

    allresp = np.where(np.deg2rad(gg['allResponse']) > np.pi/2, np.deg2rad(gg['allResponse'])-np.pi, np.deg2rad(gg['allResponse']))
    allresp = np.where(allresp < -np.pi/2, allresp+np.pi, allresp)
    allstim = np.where(np.deg2rad(gg['allTarget']) > np.pi/2, np.deg2rad(gg['allTarget'])-np.pi, np.deg2rad(gg['allTarget']))
    allstim = np.where(allstim < -np.pi/2, allstim+np.pi, allstim)

    with open('data/gam/uniform_model_base_inv_kappa_free.pkl', 'rb') as file:
        other_data = pickle.load(file)

    unif_map = usu.UnifMap(data=other_data)
    unif_map.get_cdf_and_inverse_cdf()

    # histogram coloured by rn-sn...
    binning = 31

    allresp_uni = unif_map.angle_space_to_unif_space(allresp)
    allstim_uni = unif_map.angle_space_to_unif_space(allstim)

    diff1 = allresp[1:]-allstim[1:]
    diff1 = np.where(diff1 > np.pi/2, diff1-np.pi, diff1)
    diff1 = np.where(diff1 < -np.pi/2, diff1+np.pi, diff1)

    diff = diff1

    H, xedges, yedges, binnumber = binned_statistic_2d(allstim[1:], allresp[:-1], values=diff, 
                                                statistic='mean', bins=binning, expand_binnumbers=True)
    H = np.ma.masked_where(H==0, H)
    XX, YY = np.meshgrid(xedges, yedges)

    plt.pcolormesh(XX,YY,H.T,cmap='RdBu',vmin=-0.4,vmax=0.4)
    plt.colorbar()
    plt.xlabel('$S_{n}$')
    plt.ylabel('$R_{n-1}$')
    plt.title('$\mathbb{E}[(R_{n-1} - R_n)_w + (R_{n}-S_n)_w]$')
    plt.title('$\mathbb{E}[R_{n}-S_{n}]$')
    plt.show()

    samps = np.random.permutation(allstim)
    resps = np.zeros(len(samps))
    resps[0] = samps[0]

    plt.hist(samps,bins=100)
    plt.show()

    for i in tqdm(range(1, len(samps))):
        noise = np.random.normal(loc=resps[i-1], scale=0.05)
        lookup = samps[i] + lookup_hist_value(noise,samps[i])
        #resps[i] = samps[i] + lookup_hist_value(noise,samps[i]) + np.random.normal(loc=0, scale=0.01)
        unif_lookup = unif_map.angle_space_to_unif_space(lookup)
        resps[i] = unif_map.unif_space_to_angle_space(np.random.normal(loc=unif_lookup, scale=0.1))

    samps = np.where(samps > np.pi/2, samps - np.pi, samps)
    samps = np.where(samps < -np.pi/2, samps + np.pi, samps)
    resps = np.where(resps > np.pi/2, resps - np.pi, resps)
    resps = np.where(resps < -np.pi/2, resps + np.pi, resps)

    plt.hist(resps,bins=100)
    plt.show()

    # resps_out = unif_map.unif_space_to_angle_space(resps)
    # samps_out = unif_map.unif_space_to_angle_space(samps)

    # resps_out = np.where(np.isnan(resps_out), samps_out, resps_out)

    # plt.hist(resps_out,bins=100)

    binsize = 41

    fig,ax = plt.subplots(2,2,figsize=(6,6),gridspec_kw={'height_ratios':[1, 3],'width_ratios':[3,1]})
    histo = ax[1,0].hist2d(samps, resps, bins = binsize, cmap='plasma', density=True)
    ax[1,1].barh(y=np.linspace(0,180,binsize),width=np.sum(histo[0],axis=0),height=185/binsize)
    fig.colorbar(histo[3],ax=ax[1,1])
    ax[1,1].margins(y=0)
    ax[1,1].set_yticks([])
    ax[1,1].axis('off')
    ax[0,0].bar(x=np.linspace(0,180,binsize),height=np.sum(histo[0],axis=1),width=185/binsize)
    ax[0,0].margins(x=0)
    ax[0,0].grid(False)
    ax[1,1].grid(False)
    ax[0,0].axis('off')
    ax[0,0].set_xticks([])
    ax[0,1].set_visible(False)
    # ax[1,0].set_title(title,fontsize=20,style='oblique')
    ax[1,0].set_xlabel('stimulus S',fontsize=12)
    ax[1,0].set_ylabel('response R',fontsize=12)
    histagain = histo[0]
    angspace = np.linspace(-np.pi/2,np.pi/2,binsize)
    histallrow = np.empty(binsize,dtype=list)
    for m in range(binsize):
        histallrow[m] = [int(histagain[m][k]*10e3)*[angspace[k]] for k in range(binsize)]
    hrowavg = circmean_2(histallrow,o_adjust=360/np.pi,space=1)
    histall = np.empty(binsize,dtype=list)
    for m in range(binsize):
        histall[m] = [int(histagain[k][m]*10e3)*[angspace[k]] for k in range(binsize)]
    hcolavg = circmean_2(histall,o_adjust=360/np.pi,space=1)
    hrowavg = np.where(hrowavg < -np.pi/2, hrowavg + np.pi, hrowavg)
    hrowavg = np.where(hrowavg > np.pi/2, hrowavg - np.pi, hrowavg)
    hrowavg[0] = np.where(hrowavg[0] > 0, hrowavg[0]-np.pi, hrowavg[0])
    hcolavg = np.where(hcolavg < -np.pi/2, hcolavg + np.pi, hcolavg)
    hcolavg = np.where(hcolavg > np.pi/2, hcolavg - np.pi, hcolavg)
    hcolavg[0] = np.where(hcolavg[0] > 0, hcolavg[0]-np.pi, hcolavg[0])
    ax[1,0].plot(angspace,angspace, color= 'gray')
    ax[1,0].plot(hcolavg,angspace,color='cyan',linestyle='--',label='row mean')
    ax[1,0].plot(angspace,hrowavg,color='white',linestyle='--', label='bias')
    ax[1,0].legend()
    plt.show()


def serial_dep_plots(finalsamp, finalresp, allstim, allresp, fn=None, rad = 180/np.pi):
    
    finalsamp = np.rad2deg(finalsamp)
    finalresp = np.rad2deg(finalresp)

    allstim = np.rad2deg(allstim)
    allresp = np.rad2deg(allresp)
    other = finalresp[:-1]
    # other = np.where(np.sign(allstim[1:])==np.sign(other), other, -other)
    binning = 51
    label=['$S_n$','$R_{n-1}$','$R_n$']
    fig,ax = plt.subplots(1,2,figsize=(8,5),gridspec_kw={'width_ratios':[10,1]})
    relori = np.mod((-finalsamp[1:]+other) + 3*np.pi/2*rad, np.pi*rad) - np.pi/2*rad
    #relori = allstim[1:]-np.sign(allstim[1:]*np.sign(other))*other
    relbias = np.mod(finalresp[1:]-finalsamp[1:]+3*np.pi/2*rad, np.pi*rad)-np.pi/2*rad
    bins = ax[0].hist2d(relori, relbias, bins=binning, cmap ='plasma', density=True, range=[[-90, 90], [-90, 90]])
    ax[0].set_xlabel('relative orientation, $S_n-R_{n-1}$')
    ax[0].set_ylabel('bias, '+label[2]+'$-S_n$')
    ax[0].margins(y=0)
    ax[1].set_yticks([])
    # ax[0].axis('off')
    # ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=1),width=np.pi/(binning-1))
    ax[0].margins(x=0)
    ax[0].grid(False)
    # ax[0].axis('off')
    ax[1].set_xticks([])
    ax[1].set_visible(False)
    angspace = np.linspace(-np.pi/2*rad,np.pi/2*rad,binning)
    histagain = bins[0]
    histallrow = np.empty(binning,dtype=list)
    for m in range(binning):
        histallrow[m] = [int(histagain[m][k]*10e6)*[angspace[k]] for k in range(binning)]
    hrowavg = circmean_2(histallrow,space=3, o_adjust=2)
    ax[0].plot(angspace, hrowavg, color='black', lw=1.5, linestyle= '--')
    # ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=0),height=np.pi/(binning-1))
    newbins = (bins[0]/np.sum(bins[0],axis=1)).T
    fig.colorbar(bins[3],ax=ax[0],use_gridspec=True, orientation='vertical')
    plt.show()

    other = finalresp[:-1]
    fig,ax = plt.subplots(1,2,figsize=(8,5),gridspec_kw={'width_ratios':[10,1]})
    relori = np.mod((finalsamp[1:]-other) + 3*np.pi/2*rad, np.pi*rad) - np.pi/2*rad
    #
    relori = -finalsamp[1:]+np.sign(finalsamp[1:]*np.sign(other))*other
    relbias = np.mod(finalresp[1:]-finalsamp[1:]+3*np.pi/2*rad, np.pi*rad)-np.pi/2*rad
    bins = ax[0].hist2d(relori, relbias, bins=binning, cmap ='plasma', density=True, range=[[-90, 90], [-90, 90]])
    ax[0].set_xlabel('$S_n - R_{n-1}$')
    ax[0].set_ylabel('bias, '+label[2]+'$-S_n$')
    ax[0].margins(y=0)
    ax[1].set_yticks([])
    # ax[0].axis('off')
    # ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=1),width=np.pi/(binning-1))
    ax[0].margins(x=0)
    ax[0].grid(False)
    ax[0].set_ylim([-90,90])
    # ax[0].axis('off')
    ax[1].set_xticks([])
    ax[1].set_visible(False)
    angspace = np.linspace(-np.pi/2*rad,np.pi/2*rad,binning)
    histagain = bins[0]
    histallrow = np.empty(binning,dtype=list)
    for m in range(binning):
        histallrow[m] = [int(histagain[m][k]*10e6)*[angspace[k]] for k in range(binning)]
    hrowavg2 = circmean_2(histallrow,space=3, o_adjust=2)
    ax[0].plot(angspace, hrowavg2, color='white', lw=1.5, linestyle= '--')
    # ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=0),height=np.pi/(binning-1))
    newbins = (bins[0]/np.sum(bins[0],axis=1)).T
    fig.colorbar(bins[3],ax=ax[0],use_gridspec=True, orientation='vertical')
    plt.show()

    other = finalsamp[:-1]
    fig,ax = plt.subplots(1,2,figsize=(8,5),gridspec_kw={'width_ratios':[10,1]})
    relori = np.mod((-finalsamp[1:]+other) + 3*np.pi/2*rad, np.pi*rad) - np.pi/2*rad
    #
    # relori = finalsamp[1:]-np.sign(finalsamp[1:]*np.sign(other))*other
    relbias = np.mod(finalresp[1:]-finalsamp[1:]+3*np.pi/2*rad, np.pi*rad)-np.pi/2*rad
    bins = ax[0].hist2d(relori, relbias, bins=binning, cmap ='plasma', density=True, range=[[-90, 90], [-90, 90]])
    ax[0].set_xlabel('$S_n - S_{n-1}$')
    ax[0].set_ylabel('bias, '+label[2]+'$-S_n$')
    ax[0].margins(y=0)
    ax[1].set_yticks([])
    # ax[0].axis('off')
    # ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=1),width=np.pi/(binning-1))
    ax[0].margins(x=0)
    ax[0].grid(False)
    ax[0].set_ylim([-90,90])
    # ax[0].axis('off')
    ax[1].set_xticks([])
    ax[1].set_visible(False)
    angspace = np.linspace(-np.pi/2*rad,np.pi/2*rad,binning)
    histagain = bins[0]
    histallrow = np.empty(binning,dtype=list)
    for m in range(binning):
        histallrow[m] = [int(histagain[m][k]*10e6)*[angspace[k]] for k in range(binning)]
    hrowavg6 = circmean_2(histallrow,space=3, o_adjust=2)
    ax[0].plot(angspace, hrowavg6, color='white', lw=1.5, linestyle= '--')
    # ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=0),height=np.pi/(binning-1))
    newbins = (bins[0]/np.sum(bins[0],axis=1)).T
    fig.colorbar(bins[3],ax=ax[0],use_gridspec=True, orientation='vertical')
    plt.show()

    # other = allstim[1:]
    # fig,ax = plt.subplots(1,2,figsize=(8,5),gridspec_kw={'width_ratios':[10,1]})
    # relori = np.mod((allstim[1:]-other) + 3*np.pi/2*rad, np.pi*rad) - np.pi/2*rad
    # #relori = allstim[1:]-np.sign(allstim[1:]*np.sign(other))*other
    # relbias = np.mod(allresp[1:]-allstim[1:]+3*np.pi/2*rad, np.pi*rad)-np.pi/2*rad
    # bins = ax[0].hist2d(relori, relbias, bins=binning, cmap ='plasma', density=True, vmax=3.2e-4)
    # ax[0].set_xlabel('$S_n$')
    # ax[0].set_ylabel('bias, '+label[2]+'$-S_n$')
    # ax[0].margins(y=0)
    # ax[1].set_yticks([])
    # # ax[0].axis('off')
    # # ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=1),width=np.pi/(binning-1))
    # ax[0].margins(x=0)
    # ax[0].grid(False)
    # # ax[0].axis('off')
    # ax[1].set_xticks([])
    # ax[1].set_visible(False)
    # angspace = np.linspace(-np.pi/2*rad,np.pi/2*rad,binning)
    # histagain = bins[0]
    # histallrow = np.empty(binning,dtype=list)
    # for m in range(binning):
    #     histallrow[m] = [int(histagain[m][k]*10e3)*[angspace[k]] for k in range(binning)]
    # hrowavg3 = circmean_2(histallrow,space=3, o_adjust=2)
    # ax[0].plot(angspace, hrowavg3, color='white', lw=1.5, linestyle= '--')
    # # ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=0),height=np.pi/(binning-1))
    # newbins = (bins[0]/np.sum(bins[0],axis=1)).T
    # fig.colorbar(bins[3],ax=ax[0],use_gridspec=True, orientation='vertical')
    # plt.show()

    hrowavgn = np.zeros(binning+1)
    hrowavgn[:-1] = hrowavg2
    hrowavgn[-1] = hrowavg2[0]
    hrowavgn1 = np.zeros(binning+1)
    hrowavgn1[:-1] = hrowavg
    hrowavgn1[-1] = hrowavg[0]

    rad = 180/np.pi
    other = allresp[:-1]
    # other = np.where(np.sign(allstim[1:])==np.sign(other), other, -other)
    label=['$S_n$','$R_{n-1}$','$R_n$']
    fig,ax = plt.subplots(1,2,figsize=(8,5),gridspec_kw={'width_ratios':[10,1]})
    relori = np.mod((-allstim[1:]+other) + 3*np.pi/2*rad, np.pi*rad) - np.pi/2*rad
    #relori = allstim[1:]-np.sign(allstim[1:]*np.sign(other))*other
    relbias = np.mod(allresp[1:]-allstim[1:]+3*np.pi/2*rad, np.pi*rad)-np.pi/2*rad
    bins = ax[0].hist2d(relori, relbias, bins=binning, cmap ='plasma', density=True, vmax=3.2e-4, alpha=0)
    ax[0].set_xlabel('relative orientation, $S_n-R_{n-1}$')
    ax[0].set_ylabel('bias, '+label[2]+'$-S_n$')
    ax[0].margins(y=0)
    ax[1].set_yticks([])
    # ax[0].axis('off')
    # ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=1),width=np.pi/(binning-1))
    ax[0].margins(x=0)
    ax[0].grid(False)
    # ax[0].axis('off')
    ax[1].set_xticks([])
    ax[1].set_visible(False)
    angspace = np.linspace(-np.pi/2*rad,np.pi/2*rad,binning)
    histagain = bins[0]
    histallrow = np.empty(binning,dtype=list)
    for m in range(binning):
        histallrow[m] = [int(histagain[m][k]*10e6)*[angspace[k]] for k in range(binning)]
    hrowavg5 = circmean_2(histallrow,space=3, o_adjust=2)
    ax[0].scatter(0,0,color='black')
    ax[0].scatter(25,-6,color='black')
    # ax[0].scatter(-25,6,color='black')
    ax[0].plot(angspace, hrowavg, color='black', lw=1.5, linestyle= '--',alpha=0)
    # ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=0),height=np.pi/(binning-1))
    newbins = (bins[0]/np.sum(bins[0],axis=1)).T
    fig.colorbar(bins[3],ax=ax[0],use_gridspec=True, orientation='vertical')
    plt.show()

    other = allresp[:-1]
    fig,ax = plt.subplots(1,2,figsize=(8,5),gridspec_kw={'width_ratios':[10,1]})
    relori = np.mod((allstim[1:]-other) + 3*np.pi/2*rad, np.pi*rad) - np.pi/2*rad
    relori = -allstim[1:]+np.sign(allstim[1:]*np.sign(other))*other
    relbias = np.mod(allresp[1:]-allstim[1:]+3*np.pi/2*rad, np.pi*rad)-np.pi/2*rad
    bins = ax[0].hist2d(relori, relbias, bins=binning, cmap ='plasma', density=True, vmax=3.2e-4)
    ax[0].set_xlabel('$S_n$')
    ax[0].set_ylabel('bias, '+label[2]+'$-S_n$')
    ax[0].margins(y=0)
    ax[1].set_yticks([])
    # ax[0].axis('off')
    # ax[0,0].bar(x=np.linspace(-np.pi/2,np.pi/2,binning),height=np.sum(bins[0],axis=1),width=np.pi/(binning-1))
    ax[0].margins(x=0)
    ax[0].grid(False)
    # ax[0].axis('off')
    ax[1].set_xticks([])
    ax[1].set_visible(False)
    angspace = np.linspace(-np.pi/2*rad,np.pi/2*rad,binning)
    histagain = bins[0]
    histallrow = np.empty(binning,dtype=list)
    for m in range(binning):
        histallrow[m] = [int(histagain[m][k]*10e6)*[angspace[k]] for k in range(binning)]
    hrowavg3 = circmean_2(histallrow,space=3, o_adjust=2)
    ax[0].plot(angspace, hrowavg3, color='white', lw=1.5, linestyle= '--')
    # ax[1,1].barh(y=np.linspace(-np.pi/2,np.pi/2,binning),width=np.sum(bins[0],axis=0),height=np.pi/(binning-1))
    newbins = (bins[0]/np.sum(bins[0],axis=1)).T
    fig.colorbar(bins[3],ax=ax[0],use_gridspec=True, orientation='vertical')
    plt.show()

    hrowavgn = hrowavg2
    hrowavgn1 = hrowavg
    # hrowavgn1 -= np.mean(hrowavgn1)
    # hrowavgn -= np.mean(hrowavgn)
    angspace = np.linspace(-90-45/binning,90-45/binning,binning)
    plt.plot(angspace,hrowavg5, color='black', alpha=0.4,label='true $R_{n-1}$ bias')
    plt.plot(angspace,hrowavgn1,color='black',linestyle='--',label='model $R_{n-1}$ bias')
    # plt.plot(angspace,hrowavg3, color='purple', alpha=0.4,label='true $R_{n-1}$ direction-invariant bias')
    
    
    # plt.plot(angspace,hrowavg6, color='blue', linestyle='--', label='model $S_{n-1}$ bias')
    # plt.plot(angspace,hrowavgn, color='purple', linestyle='--',label='model $R_{n-1}$ direction-invariant bias')
    

    plt.xlabel('relative orientation compared to $S_n$')
    plt.ylabel('mean bias (degrees)')
    plt.xlim(-93,93)
    plt.ylim(-12,12)
    plt.hlines(0, -90,90, color='gray', lw=1, linestyle='--')
    plt.vlines(0, -8,8, color='gray', lw=1, linestyle='--')
    plt.legend()
    plt.savefig('figs/forward/'+fn+'_relori_biases.png', dpi=150)
    plt.show()