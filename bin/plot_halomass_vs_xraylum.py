# 这里没有除以4pi 能对上，除以了nastasha的结果就高了？

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython
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
linestyles = ['-', 'dotted']
linewidths = [2,2]
linesbins = {'sumlum_fe17':[0.715,0.717],'sumlum_o7f':[0.574,0.576],'sumlum_o8':[0.653,0.656]}
natasha_res = {'sumlum_fe17':[[11.75,12.25, 12.75, 13.25, 13.75],[-1.3, -0.5, 1, 2.2, 2.5], [0.7, 1.5, 2.2, 3, 3]], 'sumlum_o7f':[[11.75,12.25, 12.75, 13.25, 13.75],[-0.5, 0, 1.2, 2.5, 2.5], [0.7, 2, 2.5, 2.7, 2.7]], 'sumlum_o8':[[11.75,12.25, 12.75, 13.25, 13.75],[-1, 0.2, 1.9, 3, 3.8], [1.2,2.1, 3.1,3.8, 4.1]]}

mass_filter = np.array([13, 13.5, 14, 14.5, 15])
fig, axs = plt.subplots(3,1, figsize = (9,22))
for j, line_type in enumerate(['sumlum_o8', 'sumlum_o7f', 'sumlum_fe17']): 
    nat_mean = (np.array(natasha_res[line_type][1]) + np.array(natasha_res[line_type][2]))/2
    axs[j].errorbar(natasha_res[line_type][0], nat_mean, xerr = 0.25, yerr = (nat_mean-natasha_res[line_type][1], natasha_res[line_type][2]-nat_mean), alpha = 0.3, color = 'k', label = 'nastasha')
    for l, prof_type in enumerate(['sph', 'cyl']):
        linestyle = linestyles[l]
        for k,part_type in enumerate(['incl', 'excl']):
            linewidth = linewidths[k]
            line = f'{line_type}_{prof_type}_{part_type}'
            
            my_med = np.zeros(mass_filter.shape); my_hierr = np.zeros(mass_filter.shape); my_loerr = np.zeros(mass_filter.shape)
            for i, mf in enumerate(mass_filter):
                print(mf)
                workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos' 
                df = pd.read_csv(f'{workpath}/xray_linelum_inr200c_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230515.csv')   
                lum = df[line] 
                # from IPython import embed
                # embed()
                my_med[i] = np.log10(np.median(lum)/np.array(linesbins[line_type]).mean()/4/np.pi/ (1.48e27)**2*1e4/1.602e-9*1e5)
                my_hierr[i] = np.log10(np.percentile(lum, 90)/np.array(linesbins[line_type]).mean()/4/np.pi/ (1.48e27)**2*1e4/1.602e-9*1e5) - np.log10(np.median(lum)/np.array(linesbins[line_type]).mean()/4/np.pi/ (1.48e27)**2*1e4/1.602e-9*1e5)
                my_loerr[i] = np.log10(np.median(lum)/np.array(linesbins[line_type]).mean()/4/np.pi/ (1.48e27)**2*1e4/1.602e-9*1e5) - np.log10(np.percentile(lum, 10)/np.array(linesbins[line_type]).mean()/4/np.pi/ (1.48e27)**2*1e4/1.602e-9*1e5)
            axs[j].errorbar(mass_filter+0.25, my_med, yerr = [my_loerr, my_hierr], xerr = np.full(mass_filter.shape, 0.25),alpha = 0.5, label = line, linewidth = linewidth, linestyle = linestyle)
            print(prof_type, part_type,my_med, my_hierr, my_loerr)

            # axs[j].set_yscale('log')
            axs[j].set_xticks(np.arange(11.5,15.5,0.5))
            # axs[j].set_ylim(1e-3, 1e7)
            axs[j].set_title(f'xray line luminosity of {line} for halos')
            axs[j].set_ylabel(f'L [photons/100ks/$\\rm m^2$]')
            axs[j].set_xlabel(f'$\\rm log_{{10}}M_{{200c}}$ [$\\rm M_\odot$]')
            axs[j].grid(True)
            axs[j].legend()
plt.savefig(f'{workpath}/../../../fig/halomass_vs_xraylum_230515.png')
