import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from glob import glob

massfilter = [13.0, 13.5]
props = ['part_masses']
reds = 0
# Define the radius bins
xbins_mean = 10**np.arange(-2,3.25,0.25)  # Mpc for 0.25 dex
xbins_med = 10**np.arange(-2,3.1,0.1)  # Mpc for 0.1 dex
xbins = {'010dex':xbins_med, '025dex':xbins_mean}

for mf in massfilter:
    for prop in props:
        # set workdir, datadir
        workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos'
        result_path = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results'
        old_data_path = f'{result_path}/results_wrong_wholeboxz_sb/xraysb_csvs_230504_{mf}_groups_1028halos'
        soap_file = glob(f'{old_data_path}/*btw*')[0]
        prof_path = f'{workpath}/xraylum_csvs_230610_{mf}_groups_radial_pkpc_cyl'
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
            fig, ax = plt.subplots()
            # read cyl mass profiles
            for binning in xbins.keys():
                for type in ['excl', 'incl']:
                    prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}_cyl.csv')
                    prof = np.array(prof)[:, :1022]
                    sum_prof_cyl = np.cumsum(prof, axis=0)
                    # print(sum_prof_cyl[:,1022:])
                    # prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}_sph.csv')
                    # sum_prof_sph = np.cumsum(prof, axis=1)
                    # m200c_mat = np.tile(m200c, (len(xbins[binning]), 1))
                    # plt.plot(xbins[binning], np.nanmedian(sum_prof_cyl/m200c_mat*1e10, axis=1), label = f'cyl_{binning}_{type}')
                    for k in range(len(sum_prof_cyl)):
                        plt.plot(xbins[binning]/r200c[k], sum_prof_cyl[:,k]/m200c[k]*1e10, label = f'cyl_{binning}_{type}', alpha = 0.2)
                    # print(np.nanmedian(prof, axis=1))
                    # print(np.nanmedian(sum_prof_cyl, axis=1))
                    # plt.plot(xbins[binning], np.mean(sum_prof_sph/m200c, axis=1), label = f'sph_{binning}_{type}')
                    # xbins_iter = np.tile(xbins[binning], len(m200c))
                    # hb_cyl = ax.hexbin(xbins_iter, sum_prof_cyl.flatten()/m200c, gridsize=30, cmap='plasma')
                    # hb_sph = ax.hexbin(xbins_iter, sum_prof_sph.flatten()/m200c, gridsize=30, cmap='plasma')
            plt.axvline(1)
            plt.axhline(1)
            plt.axhline(0.1)
            plt.xlabel('r / r200c')
            plt.ylabel('M(<r) / $M_{200c}$')
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f'gas mass fraction v.s. r')
            plt.savefig(f'{prop}_{mf}_gas-mass-frac_vs_r.png')
            print('plot has been created!')
            plt.close()
        # plot cts in bins
        if prop == 'cts':
            fig, ax = plt.subplots()
            # read cyl mass profiles
            for binning in xbins.keys():
                for type in ['excl', 'incl']:
                    prof_cyl = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}.csv')['cts']
                    # prof_sph = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}_sph.csv')['cts']
                    plt.plot(xbins[binning], np.median(prof_cyl, axis=1), label = f'cyl_{binning}_{type}')
                    # plt.plot(xbins[binning], np.mean(prof_sph, axis=1), label = f'sph_{binning}_{type}')
                    xbins_iter = np.tile(xbins[binning], len(m200c))
                    hb_cyl = ax.hexbin(xbins_iter, prof_cyl.flatten(), gridsize=30, cmap='plasma')
                    # hb_sph = ax.hexbin(xbins_iter, prof_sph.flatten(), gridsize=30, cmap='plasma')
            plt.xlabel('r (Mpc)')
            plt.ylabel('M(<r)/M(<$r_{200c}$)')
            plt.title(f'gas mass fraction v.s. r')
            plt.savefig(f'{savepath}/{prop}_{mf}_cts_vs_r.png')
            plt.close()
