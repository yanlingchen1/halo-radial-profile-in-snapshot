import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from glob import glob

# define parameters
massfilter = [13.0, 13.5]
props = ['part_masses', 'part_temperatures', 'part_dens', 'cts']
reds = 0
weighting = ['none', 'mass', 'volume']
# define plotting parameters
import seaborn as sns
cb = sns.color_palette("colorblind").as_hex()
lstyles = ['solid', 'dotted']
# Define the radius bins
xbins_mean = 10**np.arange(-2,3.25,0.25)  # Mpc for 0.25 dex
xbins_med = 10**np.arange(-2,3.1,0.1)  # Mpc for 0.1 dex
xbins = {'010dex':[xbins_med, 'med'], '025dex':[xbins_mean, 'mean']}

for mf in massfilter:
    for prop in props:
        # set workdir, datadir
        workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos'
        result_path = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results'
        old_data_path = f'{result_path}/results_wrong_wholeboxz_sb/xraysb_csvs_230504_{mf}_groups_1028halos'
        soap_file = glob(f'{old_data_path}/*btw*')[0]
        prof_path = f'{workpath}/xraylum_csvs_230612_{mf}_groups_radial_pkpc_cyl'
        savepath = f'{prof_path}/png'
        os.makedirs(savepath, exist_ok=True)
        # read M200 from soap
        halo_ids = pd.read_csv(soap_file)['halo_ids']
        # #extract halo_ids from filenames
        # filelst = glob(f'{old_data_path}/*partlum*')
        # halo_ids = [f.split('/')[-1].split('_')[3].split('halo')[-1] for f in filelst]
        # halo_ids = np.array(halo_ids).astype(int)
        with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_00{int(77-reds/0.05)}.hdf5", 'r') as catalogue_soap:
            soap_ids = np.array(catalogue_soap["VR/ID"][()])
            m200c_sp = catalogue_soap["SO/200_crit/TotalMass"][()]
            r200c_sp = catalogue_soap["SO/200_crit/SORadius"][()]

        m200c = m200c_sp[np.isin(soap_ids, halo_ids)]
        r200c = r200c_sp[np.isin(soap_ids, halo_ids)]

        
        # Plot part_mass in r versus M200c
        if prop == 'part_masses':
            for shape in ['sph', 'cyl']:
                fig, ax = plt.subplots(figsize = (8,8))
                for k, type in enumerate(['excl', 'incl']):
                    lstyle = lstyles[k]
                    for i, binning in enumerate(xbins.keys()):
                        
                            prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}_{shape}.csv')
                            # # exclude the radii bin whose cts < 50
                            # radii_cts = pd.read_csv(f'{prof_path}/cts_{binning}_{type}_{shape}.csv')
                            # prof[radii_cts<50] = np.nan
                            prof = np.array(prof)[:, :len(m200c)]
                            sum_prof = np.cumsum(prof, axis=0)
                            for k in range(len(sum_prof)):
                                if k==0:
                                    plt.plot(xbins[binning][0]/r200c[k], sum_prof[:,k]/m200c[k]*1e10, c = cb[i], label = f'{shape}_{binning}_{type}', alpha = 0.2, linestyle = lstyle)
                                else:
                                    plt.plot(xbins[binning][0]/r200c[k], sum_prof[:,k]/m200c[k]*1e10, c = cb[i], linestyle = lstyle)
                            if binning == '010dex':
                                plt.plot(xbins[binning][0]/r200c[k], np.nanmedian(sum_prof/m200c*1e10, axis=1), c = cb[i], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}', linestyle = lstyle, linewidth = 4, alpha = 0.6)
                            elif binning == '025dex':
                                plt.plot(xbins[binning][0]/r200c[k], np.nanmean(sum_prof/m200c*1e10, axis=1), c = cb[i], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}', linestyle = lstyle, linewidth = 3, alpha = 0.6)
                            else: 
                                raise ValueError('Binning type does not exist!')
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
        # plot cts in radii bins
        elif prop == 'cts':
            for shape in ['sph', 'cyl']:
                fig, ax = plt.subplots(figsize = (8,8))
                for k, type in enumerate(['excl', 'incl']):
                    lstyle = lstyles[k]
                    for i, binning in enumerate(xbins.keys()):
                            prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}_{shape}.csv')
                            # exclude the radii bin whose cts < 50
                            radii_cts = pd.read_csv(f'{prof_path}/cts_{binning}_{type}_{shape}.csv')
                            prof[radii_cts<50] = np.nan
                            prof = np.array(prof)[:, :len(m200c)]

                            for k in range(len(prof)):
                                if k == 0:
                                    plt.plot(xbins[binning][0]/r200c[k], prof[:,k], c = cb[i], label = f'{shape}_{binning}_{type}', alpha = 0.2, linestyle = lstyle)
                                else:
                                    plt.plot(xbins[binning][0]/r200c[k], prof[:,k], c = cb[i], alpha = 0.2, linestyle = lstyle)
                            if binning == '010dex':
                                plt.plot(xbins[binning][0]/r200c[k], np.nanmedian(prof, axis=1), c = cb[i], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}', linewidth = 4, linestyle = lstyle, alpha = 0.6)
                            elif binning == '025dex':
                                plt.plot(xbins[binning][0]/r200c[k], np.nanmean(prof, axis=1), c = cb[i], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}', linewidth = 3, linestyle = lstyle, alpha = 0.6)
                            else: 
                                raise ValueError('Binning type does not exist!')
                plt.xlabel('r / $r_{200c}$')
                plt.ylabel(f'{prop}')
                plt.yscale('log')
                plt.xscale('log')
                plt.title(f'{prop} in every radial bin')
                plt.legend()
                plt.savefig(f'{savepath}/{prop}_{mf}_{shape}_cts_vs_r.png')
                plt.close()
        # plot other properties in radii bins
        else:
            for shape in ['sph', 'cyl']:
                fig, ax = plt.subplots(figsize = (8,8))
                for k, type in enumerate(['excl', 'incl']):
                    lstyle = lstyles[k]
                    for i, binning in enumerate(xbins.keys()):
                    
                        prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}_{shape}.csv')
                        prof_mass = pd.read_csv(f'{prof_path}/part_masses_{binning}_{type}_{shape}.csv')
                        
                        # exclude the radii bin whose cts < 50
                        radii_cts = pd.read_csv(f'{prof_path}/cts_{binning}_{type}_{shape}.csv')
                        prof[radii_cts<50] = np.nan
                        prof = np.array(prof)[:, :len(m200c)]
                        prof_mass[radii_cts<50] = np.nan
                        prof_mass = np.array(prof_mass)[:, :len(m200c)]
                        for k in range(len(prof)):
                            if k == 0:
                                plt.plot(xbins[binning][0]/r200c[k], prof[:,k]*prof_mass[:,k]/prof_mass[:,k], c = cb[i], label = f'{shape}_{binning}_{type}', alpha = 0.2, linestyle = lstyle)
                            else:
                                plt.plot(xbins[binning][0]/r200c[k], prof[:,k]*prof_mass[:,k]/prof_mass[:,k], c = cb[i], alpha = 0.2, linestyle = lstyle)
                        if binning == '010dex':
                            plt.plot(xbins[binning][0]/r200c[k], np.nanmedian(prof*prof_mass/prof_mass, axis=1), c = cb[i], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}', linewidth = 4, linestyle = lstyle, alpha = 0.6)
                        elif binning == '025dex':
                            plt.plot(xbins[binning][0]/r200c[k], np.nanmean(prof*prof_mass/prof_mass, axis=1), c = cb[i], label = f'{shape}_{binning}_{xbins[binning][1]}_{type}', linewidth = 3, linestyle = lstyle, alpha = 0.6)
                        else: 
                            raise ValueError('Binning type does not exist!')
                plt.xlabel('r / $r_{200c}$')
                plt.ylabel(f'{prop}')
                plt.yscale('log')
                plt.xscale('log')
                plt.title(f'{prop} in every radial bin')
                plt.legend()
                plt.savefig(f'{savepath}/{prop}_{mf}_{shape}_cts_vs_r.png')
                plt.close()
