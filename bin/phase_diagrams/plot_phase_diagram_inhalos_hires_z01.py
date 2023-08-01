'''
Input: The temperatures, densities, metallicities, xraylum of particles

halos_sumfile: profiles of particles, mark the temperatures on points
nH vs r, color coded by T

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm import tqdm
from unyt import g, cm, mp
from matplotlib.colors import LogNorm
import os
import h5py

# define parameters
reds = 0
halonum = 7 # first use 50 halos to test
for mf in [13.5]:

    # define paths
    resultpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results'
    datapath = f'{resultpath}/xraysb_csvs_230607_{mf}_groups_1028halos_cyl'
    savepath = f'fig/phase_diagrams_{mf}_highres'
    os.makedirs(savepath, exist_ok=True)

    # input soap cat
    print('loading soap cat...')
    # Caution!!! only redshift 0 soap halo position is correct, for larger redshifts halo postions are wrong!

    reds = 0.1
    with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/L1000N3600/HYDRO_FIDUCIAL/SOAP/halo_properties_00{int(78-reds/0.05)}.hdf5", 'r') as catalogue_soap:
        soap_ids = np.array(catalogue_soap["VR/ID"][()])
        m200c_sp = catalogue_soap["SO/200_crit/TotalMass"][()]
        r200c_sp = catalogue_soap["SO/200_crit/SORadius"][()]
        gasmass_center = catalogue_soap["SO/200_crit/GasCentreOfMass"][()]
        # xray_bol_emilum = catalogue_soap["SO/200_crit/XRayLuminosityWithoutRecentAGNHeating"][()]
        # xray_bol_phlum = catalogue_soap["SO/200_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][()]
        
    ## for every halo
    partfilenamelist = glob(f'{datapath}/*partlum*')
    part_haloids = [f.split('/')[-1].split('_')[3].split('halo')[-1] for f in partfilenamelist]
    part_haloids = np.array(part_haloids).astype(int)

    sum_haloids = soap_ids - 1
    sum_halo_r200c = r200c_sp

    # Read halo center
    # Calculate distance from halo center
    halo_centers = gasmass_center
    for haloid in tqdm(part_haloids[:halonum]):
        part_halofilename = glob(f'{datapath}/*{haloid}*partlum*')[0]
        part_halofile = pd.read_csv(f'{part_halofilename}')
        halo_center = halo_centers[sum_haloids == haloid]
        r200c = sum_halo_r200c[sum_haloids == haloid]
        coor = np.array([part_halofile['part_xcoords'], part_halofile['part_ycoords'], part_halofile['part_zcoords']]).T
        r = np.linalg.norm(coor - halo_center, axis=1)  # Radius values
        print(np.nanmean(r), np.nanmedian(r), np.nanmin(r[r>0]), np.nanmax(r))
        
        # read densities and temperatures of the particles
        nH = part_halofile['nH_dens']  # Density values
        T = part_halofile['part_temperatures']  # Temperature 
        for line in ['o7f', 'o8', 'fe17']:
            xray = part_halofile[line]
            # print(np.nanmin(xray[xray>0]))
            # print(np.sum(T!=0))
            # print(np.sum(nH!=0))
            # Create a scatter plot with color coding
            # set 
            # filter = (nH<1e-4) & (T>1e5) & (T<1e7)
            for filter_val in [1, 2, 5, 10]:
                filter = (r/r200c)<filter_val
                plt.hexbin(nH[filter], T[filter],  reduce_C_function = np.median, bins = 'log', xscale = 'log',yscale = 'log', gridsize = 100,  cmap='viridis' )

                plt.hexbin(nH[filter], T[filter], C = np.log10(xray[filter]), reduce_C_function = np.median, bins = 'log', xscale = 'log',yscale = 'log', gridsize = 40,  cmap='viridis' )
                # plt.hexbin(nH, T, C = r/r200c, reduce_C_function = np.median, bins = 'log', xscale = 'log',yscale = 'log', gridsize = 40, norm = LogNorm(vmax = 1.5), cmap='viridis' ) # C = r/r200c, 
                # plt.hist2d(nH, T,  bins = [np.logspace(-10,2,100), np.logspace(3,9,100)], norm = LogNorm(),  cmap='viridis', weights = r/r200c) 


                plt.title(f'Particle Profiles with Markings within halo {haloid} < {filter_val} r200c')
                plt.suptitle(f'Halo mass in $10^{{{mf}}}-10^{{{mf+0.5}}} M_{{\odot}}_{filterval}r200c$ ')
                # Add colorbar and labels
                cbar = plt.colorbar()
                # cbar.set_label('log10(Radius/$R_{200c})$')
                cbar.set_label('cts')
                # cbar.set_label(f'log10(X-raylum-{line})')
                # cbar.set_label(f'r/r200c')
                # plt.axhline(1e5)
                # plt.axhline(1e7)
                # plt.axvline(1e-4)
                # plt.axvline(1e-8)
                plt.grid()
                plt.ylabel('Temperature (T)' )
                plt.xlabel('Density (nH)')
                plt.xscale('log')
                plt.yscale('log')
                # plt.savefig(f'{savepath}/{mf}_haloid{haloid}_dist_10r200c.png')
                # plt.savefig(f'{savepath}/{mf}_haloid{haloid}_xray{line}.png')
                plt.savefig(f'{savepath}/{mf}_haloid{haloid}_cts_med_in{filter_val}r200c.png')
                plt.close()
                print('figure has been saved!')