import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from glob import glob
from unyt import mp, Msun, kpc, cm

# define functions
def cal_vol(type, r_arr):
    # r_arr: np.2darray, units: kpc
    if type == 'sph':
        return np.diff(r_arr**3, axis=1) * 4/3 *np.pi
    elif type == 'cyl':
        return np.diff(r_arr**2, axis=1) * np.pi * 6.5 * 1e3
    else: 
        raise ValueError('Wrong volume input type!')
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
reds = 0.1

fig, ax = plt.subplots(figsize = (8,8))
for m, sim in enumerate(['L1000N1800', 'L1000N3600']):
    if sim == 'L1000N1800':
        snapnum = int(77-reds/0.05)
    elif sim == 'L1000N3600':
        snapnum = int(78-reds/0.05)

    weighting = ['none', 'mass', 'volume']
    # define plotting parameters
    import seaborn as sns
    cb = sns.color_palette("colorblind").as_hex()
    lstyles = ['solid', 'dotted']
    # Define the radius bins
    xbins_mean = np.arange(-1.5, 1, 0.25) # 10**xbins *r200c -> pMpc
    xbins_med = np.arange(-1.5, 1, 0.1)
    xbins = {'010dex':[xbins_med, 'med'], '025dex':[xbins_mean, 'mean']}

    for mf in massfilter:
        for prop in props:
            # set workdir, datadir
            workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/redshift_01/{sim}'
            datapath = f'{workpath}/xraysb_csvs_230718_{mf}_groups_1028halos_cyl'
            prof_path = f'{workpath}/profiles_230718_{mf}'
            savepath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/fig/profiles_230718/ind_profile'
            os.makedirs(savepath, exist_ok=True)

            # read r200c from the summary file
            print(datapath)
            sumfilename = glob(f'{datapath}/*btw*')[0]
            sumfile = pd.read_csv(sumfilename)
            halo_ids = sumfile['halo_ids']
            r200c = sumfile['r200c']
            xray_emibollum = np.array(sumfile['xray_bol_emilum'])
            
            # Plot part_mass in r versus M200c
            if prop == 'part_masses':
                for shape in ['sph']: #, 'cyl'

                    
                    for k, type in enumerate(['excl']): # , 'incl'
                        
                        for i, binning in enumerate(xbins.keys()):
                            lstyle = lstyles[i]
                            prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}_{shape}.csv')
                            # # exclude the radii bin whose cts < 50
                            # radii_cts = pd.read_csv(f'{prof_path}/cts_{binning}_{type}_{shape}.csv')
                            # prof[radii_cts<50] = np.nan
                            prof = np.array(prof)[:, :len(r200c)]
                            # for k in range(len(sum_prof)):
                            #     if k==0:
                            #         plt.plot(xbins[binning][0]/r200c[i], sum_prof[:,i]/m200c[i]*1e10, c = cb[i], label = f'{shape}_{binning}_{type}', alpha = 0.2, linestyle = lstyle)
                            #     else:
                            #         plt.plot(xbins[binning][0]/r200c[i], sum_prof[:,i]/m200c[i]*1e10, c = cb[i], linestyle = lstyle)
                            
                            
                            # weighted by xray emi bol lum in r200c from soap
                            prof = prof / xray_emibollum * np.nanmean(xray_emibollum)
                            
                            # convert solar mass/ ckpc3 to 1/ckpc3 (mp/ckpc3, i.e. how many protons per ckpc3)
                            prof = prof * (Msun/mp).value

                            if binning == '010dex':
                                #### calculate the shell volume ####
                                xs = 10 ** xbins[binning][0]
                                xs_ins = np.insert(xs,0,0)
                                xbins_ex = xs_ins * np.tile(r200c, (len(xs_ins),1)).T * 1e3
                                shell_vol = cal_vol(shape, xbins_ex)
                                plt.plot(xbins[binning][0], np.log10(np.nanmedian(prof/shell_vol.T*((1*cm).to(kpc)).value**3, axis=1)), c = cb[i], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}_{sim}', linestyle = lstyles[m], linewidth = 2, alpha = 0.8)
                            elif binning == '025dex':
                                xs = 10 ** xbins[binning][0]
                                xs_ins = np.insert(xs,0,0)
                                xbins_ex = xs_ins * np.tile(r200c, (len(xs_ins),1)).T * 1e3
                                shell_vol = cal_vol(shape, xbins_ex)
                                plt.plot(xbins[binning][0], np.log10(np.nanmedian(prof/shell_vol.T*((1*cm).to(kpc)).value**3, axis=1)), c = cb[i], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}_{sim}', linestyle = lstyles[m], linewidth = 2, alpha = 0.8)
                            else: 
                                raise ValueError('Binning type does not exist!')


plt.xlabel('$\\rm log_{10}(r / r200c)$')
plt.ylabel('$ \\rm log_{10}(M_{gas} / V_{shell})\ [1/cm^3]$ ')

plt.title(f'$\\rm 1e{mf}-{mf+0.5} M_{{\odot}}$ gas mass density v.s. r')
plt.legend()
plt.savefig(f"{savepath}/{prop}_1e{mf}_{shape}_gasmass_dens_vs_r_2sims.png")
print('plot has been created!')
plt.close()
