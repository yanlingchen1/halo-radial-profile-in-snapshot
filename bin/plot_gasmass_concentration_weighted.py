import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from glob import glob


# define functions
def filter_prof_cts(prof, thres, ctsprof, length):
    mul_prof[ctsprof<thres] = np.nan
    prof = np.array(prof)[:, :length]
    return prof


# define parameters
massfilter = [13.0, 13.5]
reds = 0
props = ['part_temperatures', 'part_dens', 'abun_hydrogen', 'abun_iron', 'abun_oxygen']
weightings = ['mass', 'vol', 'o7f', 'o8', 'fe17']

# define plotting parameters
import seaborn as sns
cb = sns.color_palette("colorblind").as_hex()
lstyles = ['solid', 'dotted']

# define the paths
workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_add_xraylum_sb_230509/'
result_path = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results'
savepath = f'/cosma8/data/dp004/dc-chen3/work/fig/profiles_230615'
os.makedirs(savepath, exist_ok=True)
# Define the radius bins
xbins_mean = 10**np.arange(-2,3.25,0.25)  # Mpc for 0.25 dex
xbins_med = 10**np.arange(-2,3.1,0.1)  # Mpc for 0.1 dex
xbins = {'010dex':[xbins_med, 'med'], '025dex':[xbins_mean, 'mean']}

for mf in massfilter:
    for prop in props:
        for weighting in weightings:
            # Plot part_mass in r versus M200c
            for shape in ['sph', 'cyl']:
                for k, type in enumerate(['excl', 'incl']):
                    # make a profile plot: {mf}_{prop}_{weighting}_{shape}_{recentpart_type}
                    fig, ax = plt.subplots(figsize = (8,8))

                    for i, binning in enumerate(xbins.keys()):
                        # define plotting style
                        lstyle = lstyles[i]
                        # define paths
                        old_data_path = f'{result_path}/results_wrong_wholeboxz_sb/xraysb_csvs_230504_{mf}_groups_1028halos'
                        prof_path = f'{workpath}/xraylum_csvs_230612_{mf}_groups_radial_pkpc_cyl'
                        # read data
                        ## read haloids from sumfile
                        sumfile = glob(f'{old_data_path}/*btw*')[0]
                        halo_ids = pd.read_csv(sumfile)['halo_ids']
                        halo_ids = np.array(halo_ids).astype(int)

                        ## from soap cat read m200c and r200c
                        with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_00{int(77-reds/0.05)}.hdf5", 'r') as catalogue_soap:
                            soap_ids = np.array(catalogue_soap["VR/ID"][()])
                            m200c_sp = catalogue_soap["SO/200_crit/TotalMass"][()]
                            r200c_sp = catalogue_soap["SO/200_crit/SORadius"][()]
                        m200c = m200c_sp[np.isin(soap_ids, halo_ids)]
                        r200c = r200c_sp[np.isin(soap_ids, halo_ids)]

                        ## read profiles
                        mul_prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_mul_{weighting}_{type}_{shape}.csv') 
                        if weighting == 'vol':
                            weight_prof = pd.read_csv(glob(f'{prof_path}/part_masses_{binning}_{type}_{shape}.csv')[0])/pd.read_csv(glob(f'{prof_path}/part_dens_{binning}_{type}_{shape}.csv')[0])
                        else:                   
                            weight_prof = pd.read_csv(glob(f'{prof_path}/*{weighting}*_{binning}_{type}_{shape}.csv')[0])

                        # exclude the radii bin whose cts < 50 
                        cts_prof = pd.read_csv(f'{prof_path}/cts_{binning}_{type}_{shape}.csv')
                        mul_prof = filter_prof_cts(mul_prof, 50, cts_prof, len(m200c))
                        weight_prof = filter_prof_cts(weight_prof, 50, cts_prof, len(m200c))

                        # weight the profile
                        final_prof = mul_prof/weight_prof
                        # plot profiles: plot profiles of all halos and the median and mean profile. only label 1 kind of line
                        for k in range(len(sum_prof)):
                            if k==0:
                                plt.plot(xbins[binning][0]/r200c[k], sum_prof[:,k]/m200c[k]*1e10, c = cb[i], label = f'{shape}_{binning}_{type}', alpha = 0.2, linestyle = lstyle)
                            else:
                                plt.plot(xbins[binning][0]/r200c[k], sum_prof[:,k]/m200c[k]*1e10, c = cb[i], linestyle = lstyle)
                            plt.plot(xbins[binning][0]/r200c[k], np.nanmedian(sum_prof/m200c*1e10, axis=1), c = cb[i], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}', linestyle = lstyle, linewidth = 4, alpha = 0.5)
                            plt.plot(xbins[binning][0]/r200c[k], np.nanmedian(sum_prof/m200c*1e10, axis=1), c = cb[i], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}', linestyle = lstyle, linewidth = 2, alpha = 0.7)

                plt.axvline(1)
                plt.axhline(1)
                plt.axhline(0.1)
                plt.xlabel('r / r200c')
                plt.ylabel('M(<r) / $M_{200c}$')
                plt.xscale('log')
                plt.yscale('log')
                plt.title(f'gas mass fraction v.s. r')
                plt.legend()
                plt.savefig(f'{savepath}/{prop}_{mf}_{shape}_gas-mass-frac_vs_r.png')
                print('plot has been created!')
                plt.close()


