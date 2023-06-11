import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from glob import glob

massfilter = [13.0, 13.5]
props = ['part_masses', 'cts']
reds = 0.1
# Define the radius bins
xbins_mean = 10**np.arange(-2,3.25,0.25)  # Mpc for 0.25 dex
xbins_med = 10**np.arange(-2,3.1,0.1)  # Mpc for 0.1 dex
xbins_names = ['010dex', '025dex']

for mf in massfilter:
    for prop in props:
        # set workdir, datadir
        workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos'
        soap_file = glob(f'{workpath}/*btw*.csv')[0]
        prof_path = f'{workpath}/xraylum_csvs_230608_{mf}_groups_radial_pkpc_cyl'
        savepath = f'{prof_path}/png'
        os.makedirs(savepath, exist_ok=True)
        # read M200 from soap

        halo_ids = pd.read_csv(soap_file)['halo_ids']
        with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/L1000N3600/HYDRO_FIDUCIAL/SOAP/halo_properties_00{int(78-reds/0.05)}.hdf5", 'r') as catalogue_soap:
            soap_ids = np.array(catalogue_soap["VR/ID"][()])
            m200c_sp = catalogue_soap["SO/200_crit/TotalMass"][()]
        m200c = m200c_sp[np.isin(soap_ids, halo_ids)]

        # read cyl mass profiles
        for binning in xbins_names:
            for type in ['excl', 'incl']:
                prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}.csv')
                sum_prof = np.cumsum(prof[''])
        # read sph mass profiles
        for binning in xbins_names:
            for type in ['excl', 'incl']:
                prof = pd.read_csv(f'{prof_path}/{prop}_{binning}_{type}_sph.csv')
                sum_prof = 
        # make plot

        # save plot
