import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


cb = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']
linesbins = {'fe17':[0.715,0.717],'o7f':[0.574,0.576],'o8':[0.653,0.656]}

natasha_res = {'fe17':[[11.75,12.25, 12.75, 13.25, 13.75],[-1,2.3,2.7,3,3]], 'o7f':[[11.75,12.25, 12.75, 13.25, 13.75],[0,1,2,3,3]], 'o8':[[11.75,12.25, 12.75, 13.25, 13.75],[0.5,1.3,2.8,3.5,4]]}
mass_filter = np.array([13, 13.5, 14, 14.5, 15])
fig, axs = plt.subplots(3,1, figsize = (6,18), sharex = True)
for i, mf in enumerate(mass_filter):
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/xraysb_csvs_230504_13.0_groups_1028halos' 
    df = pd.read_csv(f'{workpath}/xray_linelum_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230404.csv') 
    haloids = df['halo_ids']
    for j, line in enumerate(['o8', 'o7f', 'fe17']): 
        lum = np.full((len(haloids)), np.nan)
        for k, haloid in enumerate(haloids):
            df_part = pd.read_csv(f'{workpath}/xray_linelum_snapshot75_halo{int(haloid-1)}_partlum_230404.csv')
            lum[k] = np.nansum(df_part[line][df_part['jointmsk']])
        axs[j].errorbar(np.ones(len(lum))*(mf+0.25), np.median(lum, axis=1)/np.array(linesbins[line]).mean()/(3.09e24)**2*1e4/1.602e-9*1e5, xerr = 0.25,color = cb[i], fmt = '.')
        axs[j].errorbar(np.ones(len(lum))*(mf+0.25), np.percentile(lum,84, axis=1)/np.array(linesbins[line]).mean()/(3.09e24)**2*1e4/1.602e-9*1e5, xerr = 0.25,color = cb[i], fmt = '.')
        axs[j].errorbar(np.ones(len(lum))*(mf+0.25), np.percentile(lum,16, axis=1)/np.array(linesbins[line]).mean()/(3.09e24)**2*1e4/1.602e-9*1e5, xerr = 0.25,color = cb[i], fmt = '.')
for j, line in enumerate(['o8', 'o7f', 'fe17']):
    for k in range(2):
        axs[j].errorbar(natasha_res[line][0], np.power(10,natasha_res[line][1]), xerr = 0.25, alpha = 0.1, color = 'k')
    axs[j].set_yscale('log')
    axs[j].set_xticks(np.arange(11.5,15.5,0.5))
    axs[j].set_ylim(1e-3, 1e7)
    axs[j].set_title(f'xray line luminosity of {line} for halos')
    axs[j].set_ylabel(f'L [photons/100ks/$\\rm m^2$]')
    axs[j].set_xlabel(f'$\\rm log_{{10}}M_{{200c}}$ [$\\rm M_\odot$]')
    axs[j].grid(True)
plt.savefig(f'{workpath}/png/halomass_vs_xraylum.png')
