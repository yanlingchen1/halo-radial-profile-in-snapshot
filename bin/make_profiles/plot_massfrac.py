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
massfilter = [13.0]
props = ['part_masses']
# props = ['o7f', 'o8', 'fe17']#['part_masses', 'part_temperatures', 'part_dens', 'cts', ]
# define sims

reds = 0.1

sim = 'L1000N1800'
snapnum = int(77-reds/0.05)

# sim = 'L1000N3600'
# snapnum = int(78-reds/0.05)

# define plotting parameters
import seaborn as sns
cb = sns.color_palette("colorblind").as_hex()
lstyles = ['solid', 'dotted']
# Define the radius bins
xbins_mean = np.arange(-1.5, 1, 0.25) # 10**xbins *r500c -> pMpc
xbins_med = np.arange(-1.5, 1, 0.1)

xbins = {'010dex':[xbins_med, 'med']} # , 

for mf in massfilter:
    for prop in props:
        # set workdir, datadir
        workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/redshift_01/{sim}'
        datapath = f'{workpath}/xraysb_csvs_230718_{mf}_groups_128halos_cyl'
        prof_path = f'{workpath}/profiles_230718_{mf}_ind_r500c'
        savepath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/fig/profiles_230805/ind_profile'
        os.makedirs(savepath, exist_ok=True)        

        # read r500c from the summary file
        sumfilename = glob(f'{datapath}/*btw*')[0]
        sumfile = pd.read_csv(sumfilename)
        halo_ids = sumfile['halo_ids']
        # r500c = sumfile['r500c']

        # read total mass M500c in soap cat
        with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/{sim}/HYDRO_FIDUCIAL/SOAP/halo_properties_00{snapnum}.hdf5", 'r') as catalogue_soap:
            soap_ids = np.array(catalogue_soap["VR/ID"][()])
            m500c_sp = catalogue_soap["SO/500_crit/TotalMass"][()]
            r500c_sp = catalogue_soap["SO/500_crit/SORadius"][()] * (1+reds)

        m500c = m500c_sp[halo_ids-1]/1e10
        r500c = r500c_sp[halo_ids-1]

        # Plot part_mass in r versus M500c
        if prop == 'part_masses':
            for shape in ['sph']: #, 'cyl'
                fig, ax = plt.subplots(figsize = (8,8))
                for k, type in enumerate(['rhp']):
                    for i, binning in enumerate(xbins.keys()):
                        lstyle = lstyles[i]
                        prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}_{shape}.csv')
                        # # exclude the radii bin whose cts < 50
                        # radii_cts = pd.read_csv(f'{prof_path}/cts_{binning}_{type}_{shape}.csv')
                        # prof[radii_cts<50] = np.nan
                        prof = np.array(prof)[:, :len(r500c)]
                        sum_prof = np.cumsum(prof, axis=0)

                        if binning == '010dex':
                            ### choose the sum mas enclosed in r500c (closed to the med(r500c) of all halos as the r500c mass)
                            plt.plot(xbins[binning][0], np.nanmedian(sum_prof/m500c, axis=1), c = cb[k], label = f'{shape}_{binning}_median_{type}', linewidth = 2, alpha = 0.8)
                        elif binning == '025dex':
                            plt.plot(xbins[binning][0], np.nanmedian(sum_prof/m500c, axis=1), c = cb[k], label = f'{shape}_{binning}_median_{type}', linewidth = 2, alpha = 0.8)
                        else: 
                            raise ValueError('Binning type does not exist!')

                plt.xlabel('$\\rm log_{10}(r / r_{500c})$')
                plt.ylabel('$\\rm M_{gas}(<r)$ / $\\rm M_{tot}(<r500c)$')
                plt.title(f'$\\rm 1e{mf}-{mf+0.5} M_{{\odot}}$ gas mass fraction v.s. r')
                plt.legend()
                plt.savefig(f"{savepath}/{prop}_1e{mf}_{shape}_gas-mass_vs_totgasmass-frac_vs_r_inr500c_NOTweightedby_emilum_inr500c.png")
                print('plot has been created!')
                plt.close()
