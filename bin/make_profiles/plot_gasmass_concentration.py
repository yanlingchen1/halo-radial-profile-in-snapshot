import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from glob import glob

# define functions
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

# define parameters
massfilter = [13.5]
props = ['part_masses']
# props = ['o7f', 'o8', 'fe17']#['part_masses', 'part_temperatures', 'part_dens', 'cts', ]
# define sims

reds = 0.1

#create the plot
fig, ax = plt.subplots(figsize = (8,8))


for m, sim in enumerate(['L1000N1800']): # , 'L1000N3600'
    if sim == 'L1000N1800':
        snapnum = int(77-reds/0.05)
    elif sim == 'L1000N3600':
        snapnum = int(78-reds/0.05)

    # define plotting parameters
    import seaborn as sns
    cb = sns.color_palette("colorblind").as_hex()
    lstyles = ['solid', 'dotted']
    # Define the radius bins
    xbins_mean = np.arange(-1.5, 1, 0.25) # 10**xbins *r500c -> pMpc
    xbins_med = np.arange(-1.5, 1, 0.1)

    xbins = {'010dex':[xbins_med, 'med']} # , '025dex':[xbins_mean, 'mean']

    for mf in massfilter:
        for prop in props:
            # set workdir, datadir
            workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/redshift_01/{sim}'
            datapath = f'{workpath}/xraysb_csvs_230718_{mf}_groups_1028halos_cyl'
            prof_path = f'{workpath}/profiles_230718_{mf}'
            savepath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/fig/profiles_230718/ind_profile'
            os.makedirs(savepath, exist_ok=True)        

            # read r500c from the summary file
            sumfilename = glob(f'{datapath}/*btw*')[0]
            sumfile = pd.read_csv(sumfilename)
            halo_ids = sumfile['halo_ids']
            r200c = sumfile['r200c']
            xray_emibollum = np.array(sumfile['xray_bol_emilum'])

            
            # Plot part_mass in r versus M500c
            if prop == 'part_masses':
                for shape in ['sph']: #, 'cyl'
                    
                    for k, type in enumerate(['excl', 'incl']): #, 
                        
                        for i, binning in enumerate(xbins.keys()):
                            lstyle = lstyles[i]
                            prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}_{shape}.csv')
                            # # exclude the radii bin whose cts < 50
                            # radii_cts = pd.read_csv(f'{prof_path}/cts_{binning}_{type}_{shape}.csv')
                            # prof[radii_cts<50] = np.nan
                            prof = np.array(prof)[:, :len(r200c)]
                            sum_prof = np.cumsum(prof, axis=0)

                            # weighted by xray emi bol lum in r200c from soap
                            sum_prof = sum_prof / xray_emibollum * np.nanmean(xray_emibollum)

                            # for k in range(len(sum_prof)):
                            #     if k==0:
                            #         plt.plot(xbins[binning][0]/r500c[i], sum_prof[:,i]/m500c[i]*1e10, c = cb[i], label = f'{shape}_{binning}_{type}', alpha = 0.2, linestyle = lstyle)
                            #     else:
                            #         plt.plot(xbins[binning][0]/r500c[i], sum_prof[:,i]/m500c[i]*1e10, c = cb[i], linestyle = lstyle)
                            if binning == '010dex':
                                ### choose the sum mas enclosed in r500c (closed to the med(r500c) of all halos as the r500c mass)
                                msk = np.argmin(abs(xbins[binning][0]))

                                plt.plot(xbins[binning][0], np.log10(np.nanmedian(sum_prof/sum_prof[msk], axis=1)), c = cb[0], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}_{sim}', linestyle = lstyles[k], linewidth = 2, alpha = 0.8)
                            elif binning == '025dex':
                                msk = np.argmin(abs(xbins[binning][0]))
                                plt.plot(xbins[binning][0], np.log10(np.nanmean(sum_prof/sum_prof[msk], axis=1)), c = cb[1], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}_{sim}', linestyle = lstyles[k], linewidth = 2, alpha = 0.8)
                            else: 
                                raise ValueError('Binning type does not exist!')

### plot nastasha data ###


nas_dir = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/nastasha_plots_data'
df = pd.read_csv(f'{nas_dir}/gasmassfrac_halomass135-140.csv', header=None)
plt.plot(df[0], df[1], label = 'Nastasha', c = cb[-2])
plt.xlabel('$\\rm log_{10}(r / r_{200c})$')
plt.ylabel('$\\rm log_{10}(M_{gas}(<r)$ / $\\rm M_{gas}(<r200c))$')

plt.title(f'$\\rm 1e{mf}-{mf+0.5} M_{{\odot}}$ gas mass concentration v.s. r')
plt.legend()
plt.savefig(f"{savepath}/{prop}_1e{mf}_{shape}_gas-mass_concentration-frac_vs_r_in_r500c.png")
print('plot has been created!')
plt.close()


