import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from glob import glob
from datetime import datetime


### The densities (nH or particles) are linear, not in log!
# define functions
def filter_prof_cts(prof, thres, ctsprof, length):
    prof[ctsprof<thres] = np.nan
    prof = np.array(prof)[:, :length]
    return prof

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
massfilter = [13.0, 13.5]
reds = 0
props = ['part_temperatures', 'nH_dens', 'abun_hydrogen', 'abun_iron', 'abun_oxygen']
units = ['K', '$\\rm cm^{-3}$', '$Z_{\odot}$']
weightings = ['mass', 'o7f', 'o8', 'fe17'] # 'vol', 

# define plotting parameters
import seaborn as sns
cb = sns.color_palette("colorblind").as_hex()
lstyles = ['solid', 'dotted']

# define the paths
workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_add_xraylum_sb_230509'
result_path = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results'
savepath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/fig/profiles_230615'
os.makedirs(savepath, exist_ok=True)
# Define the radius bins
xbins_mean = 10**np.arange(-2,3.25,0.25)  # Mpc for 0.25 dex
xbins_med = 10**np.arange(-2,3.1,0.1)  # Mpc for 0.1 dex
xbins = {'010dex':[xbins_med, 'med'], '025dex':[xbins_mean, 'mean']}

for mf in massfilter:
    print(mf)
    for i, prop in enumerate(props):
        print(prop)
        for weighting in weightings:
            print(weighting)
            for shape in ['sph', 'cyl']:
                print(prop)
                for k, type in enumerate(['excl', 'incl']):
                    print(type)
                    # make a profile plot: {mf}_{prop}_{weighting}_{shape}_{recentpart_type}
                    fig, ax = plt.subplots(figsize = (8,8))

                    for j, binning in enumerate(xbins.keys()):
                        # define plotting style
                        lstyle = lstyles[j]
                        # define paths
                        old_data_path = f'{result_path}/results_wrong_wholeboxz_sb/xraysb_csvs_230504_{mf}_groups_1028halos'
                        prof_path = f'{workpath}/xraysb_csvs_{mf}_groups_1028halos/xraylum_csvs_230612_{mf}_groups_radial_pkpc_cyl'
                        # read data
                        ## read haloids from sumfile
                        sumfile = glob(f'{old_data_path}/*btw*')[0]
                        halo_ids = pd.read_csv(sumfile)['halo_ids']
                        halo_ids = np.array(halo_ids).astype(int)

                        ## from soap cat read m200c and r200c
                        with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_00{int(77-reds/0.05)}.hdf5", 'r') as catalogue_soap:
                            soap_ids = np.array(catalogue_soap["VR/ID"][()])
                            r200c_sp = catalogue_soap["SO/200_crit/SORadius"][()]
                        r200c = r200c_sp[np.isin(soap_ids, halo_ids)]

                        ## read profiles
                        mul_prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_mul_{weighting}_{type}_mul_{shape}.csv') 
                        print(f'{prop}-{weighting}:{np.nanmin(mul_prof[mul_prof>0]), np.nanmax(mul_prof), np.nanmedian(mul_prof)}')
                    #     if weighting == 'vol':
                    #         mass_profname = f'{prof_path}/part_masses_{binning}_{type}_{shape}.csv'
                    #         dens_profname = f'{prof_path}/part_dens_{binning}_{type}_{shape}.csv'
                    #         weight_prof = pd.read_csv(mass_profname)/pd.read_csv(dens_profname)
                    #     elif weighting == 'mass':                   
                    #         weight_prof = pd.read_csv(f'{prof_path}/part_masses_{binning}_{type}_{shape}.csv')
                    #     else:
                    #         weight_prof = pd.read_csv(glob(f'{prof_path}/{weighting}_{binning}_{type}_{shape}.csv')[0])

                    #     # exclude the radii bin whose cts < 50 and cut the data of zeros rows. 
                    #     cts_prof = pd.read_csv(f'{prof_path}/cts_{binning}_{type}_{shape}.csv')
                    #     mul_prof = filter_prof_cts(mul_prof, 50, cts_prof, len(r200c))
                    #     weight_prof = filter_prof_cts(weight_prof, 50, cts_prof, len(r200c))

                    #     # weight the profile
                    #     final_prof = mul_prof/weight_prof
                    #     # plot profiles: plot profiles of all halos and the median and mean profile. only label 1 kind of line
                    #     for k in range(len(final_prof)):
                    #         if k==0:
                    #             plt.plot(xbins[binning][0]/r200c[k], final_prof[:,k], c = cb[j], label = f'{shape}_{binning}_{type}', alpha = 0.1, linestyle = lstyle)
                    #         else:
                    #             plt.plot(xbins[binning][0]/r200c[k], final_prof[:,k], c = cb[j], linestyle = lstyle, alpha = 0.1)
                    #     plt.plot(xbins[binning][0]/r200c[k], np.nanmedian(final_prof, axis=1), c = cb[j], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}', linestyle = lstyle, linewidth = 3, alpha = 0.7)
                    #     plt.plot(xbins[binning][0]/r200c[k], np.nanmedian(final_prof, axis=1), c = cb[j], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}', linestyle = lstyle, linewidth = 3, alpha = 0.8)
                        
                    #     # plot relative abundance


                    # plt.xlabel('r / r200c')
                    # plt.ylabel(f'{prop} weighted by {weighting} [{units[i]}]')
                    # plt.xscale('log')
                    # plt.yscale('log')
                    # plt.title(f'1000 halos in halo mass bin $10^{{{mf}}}$ - $10^{{{mf+0.5}}}$ at z=0 \n {shape}-{type}')
                    # plt.legend()
                    # plt.savefig(f'{savepath}/z0_{mf}_{prop}_weightby_{weighting}_{shape}_{type}.png')
                    # print(f'{datetime.now()}:plot has been created!')
                    # plt.close()


