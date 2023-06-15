'''
Question:
check why my median and mean profile has 1 order magnitude discrepancy

Possible answer:
my sample are biased to low mass halo

Procedures:
1. plot all halo mass distributions in particular mass bin in soap
2. plot my sample's halo mass distribution
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import h5py
import os

for mf in [13.0, 13.5]:
    # define paths
    resultpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_wrong_wholeboxz_sb'
    datapath = f'{resultpath}/xraysb_csvs_230504_{mf}_groups_1028halos'
    savepath = f'../fig/halomass_distribution'
    os.makedirs(savepath, exist_ok = True)
    # read data
    sumfilename = glob(f'{datapath}/*btw*')[0]
    sumfile = pd.read_csv(sumfilename)
    ## for every halo extract the halo ids
    partfilenamelist = glob(f'{datapath}/*partlum*')
    part_haloids = [f.split('/')[-1].split('_')[3].split('halo')[-1] for f in partfilenamelist]
    part_haloids = np.array(part_haloids).astype(int) + 1

    # read soap
    reds = 0
    with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_00{int(77-reds/0.05)}.hdf5", 'r') as catalogue_soap:
        soap_ids = np.array(catalogue_soap["VR/ID"][()]) 
        m200c_sp_gm = catalogue_soap["SO/200_crit/GasMass"][()]
        m200c_sp_tm = catalogue_soap["SO/200_crit/TotalMass"][()]

    # plot gas mass    
    m200c_sp = m200c_sp_gm
    m200c_sim = m200c_sp[np.isin(np.array(soap_ids).astype(int), part_haloids)]
    print(np.sum(np.isin(np.array(soap_ids).astype(int), part_haloids)))
    print(np.nanmin(m200c_sp[m200c_sp>0]), np.nanmax(m200c_sp), np.nanmedian(m200c_sp))
    print(np.nanmin(m200c_sim[m200c_sim>0]), np.nanmax(m200c_sim), np.nanmedian(m200c_sim))
    plt.hist(m200c_sp, np.logspace(0, 16, 300), alpha = 0.3, label = 'all halos')
    plt.hist(m200c_sim, np.logspace(0, 16, 300), alpha = 0.3, label = 'selected halos')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('halo counts')
    plt.xlabel('halo gas mass in r200c')
    plt.legend()
    plt.savefig(f'{savepath}/flamingo_medres_1Gpc_gasmass_{mf}.png')
    plt.close()
    # plot total mass
    m200c_sp = m200c_sp_tm
    m200c_sim = m200c_sp[np.isin(np.array(soap_ids).astype(int), part_haloids)]
    plt.hist(m200c_sp[m200c_sp>0], np.logspace(0, 16, 300), alpha = 0.3, label = 'all halos')
    plt.hist(m200c_sim[m200c_sim>0], np.logspace(0, 16, 300), alpha = 0.3, label = 'selected halos')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('halo counts')
    plt.xlabel('halo gas mass in r200c')
    plt.legend()
    plt.savefig(f'{savepath}/flamingo_medres_1Gpc_totmass_{mf}.png')
    plt.close()
    plt.scatter(m200c_sim)
    # # plot soap filter test
    # filter = (m200c_sp_tm < np.power(10, mf))
    # soap_ids_fil = soap_ids[filter]
    # plt.hist(m200c_sp_tm[np.isin(soap_ids, soap_ids_fil)], np.logspace(0, 16, 300), alpha = 0.3, label = 'selected halos')
    # plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.ylabel('halo counts')
    # plt.xlabel('halo gas mass in r200c')
    # plt.savefig(f'{savepath}/tst_{mf}.png')
    # plt.close()