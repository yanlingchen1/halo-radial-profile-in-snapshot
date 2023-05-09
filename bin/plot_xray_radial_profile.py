import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os 

def plot_doubley(xlab, ylab, y2lab, y1tick,  y2tick):
    # Create the figure and the first axis
    fig, ax1 = plt.subplots(figsize = (8,8))
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)

    # Create the second y-axis and set it to be identical to the first
    ax2 = ax1.twinx()
    ax2.set_ylabel(y2lab)
    ax1.set_yticks(y1tick)
    ax2.set_yticks(y2tick)
    return fig, ax1

def basic_figure_style():
    SMALL_SIZE = 6*2                                    
    MEDIUM_SIZE = 8*2
    BIGGER_SIZE = 10*2

    plt.rc('font', size=MEDIUM_SIZE, family='serif')          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)                     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)                    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE, direction='out')    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE, direction='out')    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)                    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)                  # fontsize of the figure title
    plt.rc('lines', linewidth=2) 
    plt.rc('axes', grid=True) #
    plt.rc('grid', alpha=0.7) #
    plt.rc('xtick', top=True)
    plt.rc('ytick', right=True)
    plt.rc('axes.formatter', use_mathtext=True, min_exponent=4, useoffset=False)


    # plt.rc('figure', figsize='8, 6')                         # size of the figure, used to be '4, 3' in inches
    ######################################################
basic_figure_style()
cb = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']

# mf = 13.5
# nat_med = {'fe17':[(10,3),(100, 2.6),(300, 2.5),(500, 2),(800, 1), (1000,0), (1500, -1)], 'o8':[(10, 4),(50, 4),(100, 3.6),(200,3.5),(500, 3),(800, 2), (1000, 1), (1500,0),(2000, -0.5)], 'o7f':[(10, 3),(30, 3),(50, 2.5),(100, 2),(300,1.8),(500,1.6),(800, 1.5), (1000, 0.5), (2000, -0.5)]}
# nat_mean = {'fe17':[(10,4),(100, 3),(300, 2.5),(500, 2), (1000,1), (1500, 0)], 'o8':[(10, 4.8),(50, 4.9), (100, 3.9),(200,3.6),(500, 3),(800, 2), (1000, 1.8), (1500,1.5),(2000, 1)], 'o7f':[(10, 4),(30, 4),(50, 3.5),(100, 2.5),(300,1.8),(500,1.6),(800, 1.5), (1000, 1), (2000, 0.5)]}

mf = 13.0
nat_med = {'o7f':[(10,2.8),(50, 2.8),(100, 2.1),(200, 2.1),(500, 2.1), (800,1), (1000, -1)], 'fe17':[(10, 2.9),(50, 2.85),(100, 2.6),(200,2.5),(500, 2),(800, 1), (950, -1)], 'o8':[(10, 3.5),(30, 3.5),(50, 3.4),(100, 3.1),(200,3),(400,2.5),(500,2),(800, 1), (1000, 0), (1800, -1)]}
nat_mean = {'o7f':[(10,3.5),(50, 3),(100, 2.5),(200, 2.1),(500, 2.1), (800,1), (1000, 0.5)], 'fe17':[(10, 3.5),(50, 3),(100, 2.8),(200,2.6),(500, 2),(800, 1), (950, 0.2)], 'o8':[(10, 4),(30, 3.9),(50, 3.8),(100, 3.3),(200,3),(400,2.5),(500,2),(800, 1), (1000, 0.9), (1800, 0.8)]}

# mf = 14.0
# xbins = np.linspace(-1.3,5,200)
xbins = np.linspace(-2,3.1,50)

workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/xraysb_csvs_230504_{mf}_groups_1028halos/xraylum_csvs_230508_{mf}_groups_radial_pkpc_cylinder'
savepath = workpath
propfile = pd.read_csv(f'{workpath}/../xray_linelum_snapshot75_halomass_btw_{int(mf*10)}_{int(mf*10+5)}_230404.csv')
r200c = propfile['r200c']

linestyle = ['-', '--', '--']
# new linesbins
linesbins = {'fe17':[0.724, 0.726],'o7f':[0.574,0.576],'o8':[0.653,0.656]}
def unit2wij_ri(value, mode, bins): # bins unit Mpc
    # last column of value is 0
    return np.log10(value[:-1]/np.array(linesbins[mode]).mean()/3.09e24**2*1e4/1.602e-9*1e5/(np.diff(bins**2)*3.14*1e6)*((204*(1+0.1))**2*3.14))
def unit2wij_le(value, mode, bins):
    return np.log10(value[:-1]/np.array(linesbins[mode]).mean()/3.09e24**2/1.602e-9/(np.diff(bins**2)*3.14*1e6)*((204*(1+0.1))**2*3.14)/8.46e-7)
def convert_ri_to_le(y):
    return np.log10(np.power(10,y)/1e4/1e5 * 1.1818e6)
def convert_le_to_ri(y):
    return np.log10(np.power(10,y)*1e4*1e5 /1.1818e6)
# plt.ylabel(f'$\\rm log_{{10}}$SB [photons/100ks/$\\rm m^2/10arcmin^2$]')
# plt.xlabel(f'$\\rm log_{{10}}$r/r200c')
for i, mode in enumerate(['fe17', 'o7f', 'o8']):
    fig, ax1 = plot_doubley(f'$\\rm pkpc$',  f'$\\rm log_{{10}}$SB [photons/s/$\\rm cm^2/sr$]',f'$\\rm log_{{10}}$SB [photons/100ks/$\\rm m^2/10arcmin^2$]', np.arange(-4.0,4.0,1.0), np.ceil(convert_le_to_ri(np.arange(-4.0,4.0,1.0))))
    dat = pd.read_csv(f'{workpath}/{mode}.csv')
    dat[~np.isfinite(dat)] = 0
    prop_cts = np.sum(dat>80, axis=1)[:-1]
    cts_msk = (prop_cts/max(prop_cts))>0.9
    # print(prop_cts)
    # print(np.sum(cts_msk))
    bins = np.power(10,xbins)
    bins_mid = (bins[0:-1] +bins[1:])/2
    prop_med = unit2wij_le(np.nanmedian(dat, axis=1), mode, bins)
    prop_hi = unit2wij_le(np.percentile(dat, 84, axis=1), mode, bins)
    prop_lo = unit2wij_le(np.percentile(dat, 16, axis=1), mode, bins)
    prop_mean = unit2wij_le(np.nanmean(dat, axis=1), mode, bins)
    doc  = {}
    doc['med']= prop_med
    doc['hi'] = prop_hi
    doc['lo'] = prop_lo
    doc['mean'] = prop_mean
    # bins = bins
    df = pd.DataFrame.from_dict(doc)
    os.makedirs(f'{savepath}/png/', exist_ok=True)
    df.to_csv(f'{savepath}/png/halomass{int(mf*10)}-{int((mf+0.5)*10)}_xraylum_medians_{mode}.csv')  
    x_nat = np.array(nat_med[mode])[:,0]#/np.nanmedian(r200c)/1000
    y_nat = convert_ri_to_le(np.array(nat_med[mode])[:,1])
    ax1.plot(x_nat, y_nat, label = f'nastasha_med_{mode}', linestyle = '-', linewidth = 5, c = cb[i], alpha = 0.6)
    x_nat = np.array(nat_mean[mode])[:,0]#/np.nanmedian(r200c)/1000
    y_nat = convert_ri_to_le(np.array(nat_mean[mode])[:,1])
    ax1.plot(x_nat, y_nat, label = f'nastasha_mean_{mode}', linestyle = 'dotted', linewidth = 5, c = cb[i], alpha = 0.6)
    ax1.plot(bins_mid[cts_msk]*1000, prop_med[cts_msk],c = cb[i], label = f'{mode}_med')
    ax1.plot(bins_mid*1000, prop_mean,c = cb[i], linestyle = 'dotted', label = f'{mode}_mean')
    # ax1.fill_between(bins_mid[cts_msk]*1000, prop_lo[cts_msk], prop_hi[cts_msk], color = cb[i], alpha = 0.3)
    ax1.set_xlim(10,5000)
    ax1.set_ylim(-3.1,2)
    ax1.legend()
    ax1.set_xscale('log')
    plt.suptitle(f'halomass $10^{{{mf:.1f}-{(mf+0.5):.1f}}}$ solarmass')
    plt.savefig(f'{workpath}/png/halomass{int(mf*10)}-{int((mf+0.5)*10)}_{mode}_xraylum_medians.png')
    plt.close()