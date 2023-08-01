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

def msk_in_sph(coor, halo_center, r1, r2):
    dists = np.linalg.norm(coor-halo_center, axis=1)
    where = (dists< r2) & (dists>=r1)
    return where

# define parameters
reds = 0.1
halonum = 2 # first use 50 halos to test
for mf in [13.5]:

    # define paths
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot'
    resultpath = f'{workpath}/results/redshift_01/L1000N1800/xraysb_csvs_230718_13.5_groups_1028halos_cyl'
    newdatapath = f'{resultpath}'
    savepath = f'{workpath}/fig/phase_diagrams_230724/{mf}'
    os.makedirs(savepath, exist_ok=True)
    olddatapath = f'{resultpath}'

    # Input data
    halos_sumfilename = glob(f'{olddatapath}/*btw*')[0]

    ## for every halo
    partfilenamelist = glob(f'{olddatapath}/*partlum*')
    part_haloids = [f.split('/')[-1].split('_')[3].split('halo')[-1] for f in partfilenamelist]
    part_haloids = np.array(part_haloids).astype(int)

    halos_sumfile = pd.read_csv(halos_sumfilename)
    sum_haloids = halos_sumfile['halo_ids'][:len(part_haloids)] - 1
    sum_halo_r200c = np.array(halos_sumfile['r200c'])[:len(part_haloids)]
    # Read halo center
    # Calculate distance from halo center
    halo_centers = np.array([halos_sumfile['x_gasmass_center'],  halos_sumfile['y_gasmass_center'], halos_sumfile['z_gasmass_center']]).T
    halo_centers = halo_centers[:len(part_haloids)]
    for haloid in tqdm(part_haloids[:halonum]):
        part_halofilename = glob(f'{olddatapath}/*{haloid}*partlum*')[0]
        part_halofile = pd.read_csv(f'{part_halofilename}')
        halo_center = halo_centers[sum_haloids == haloid]
        r200c = sum_halo_r200c[sum_haloids == haloid]
        coor = np.array([part_halofile['part_ycoords'], part_halofile['part_zcoords'], part_halofile['tot_abun_to_solar']]).T
        r = np.linalg.norm(coor - halo_center, axis=1)  # Radius values
        print(np.nanmean(r), np.nanmedian(r), np.nanmin(r[r>0]), np.nanmax(r))

        for line in ['o7f', 'o8', 'fe17']:
            # read densities and temperatures of the particles
            nH = part_halofile['nH_dens']  # Density values
            T = part_halofile['part_temperatures']  # Temperature 
            xray = part_halofile[line]
            # print(np.nanmin(xray[xray>0]))
            # print(np.sum(T!=0))
            # print(np.sum(nH!=0))
            # Create a scatter plot with color coding

            # set the spatial filter
            upper_r = 1
            lower_r = 0
            filter = (r/r200c<upper_r) & (r/r200c>lower_r)
            fig, ax = plt.subplots()
            plt.hexbin(nH[filter], T[filter],  C=np.log10(xray[filter]), reduce_C_function = np.median, bins = 'log', xscale = 'log',yscale = 'log', gridsize = 40,  cmap='viridis' )
            # plt.hexbin(nH, T, C = r/r200c, reduce_C_function = np.median, bins = 'log', xscale = 'log',yscale = 'log', gridsize = 40, norm = LogNorm(vmax = 1.5), cmap='viridis' ) # C = r/r200c, 
            # plt.hist2d(nH, T,  bins = [np.logspace(-10,2,100), np.logspace(3,9,100)], norm = LogNorm(),  cmap='viridis', weights = r/r200c) 


            plt.title(f'Halo mass in $10^{{{mf}}}-10^{{{mf+0.5}}} M_{{\odot}}$')
            plt.suptitle(f' in {lower_r}r_{{200c}} - {upper_r} r_{{200c}}')
            # Add colorbar and labels
            cbar = plt.colorbar()
            # cbar.set_label('log10(Radius/$R_{200c})$')
            # cbar.set_label('cts')
            cbar.set_label(f'X-raylum-{line}')
            # plt.axhline(1e5)
            # plt.axhline(1e7)
            # plt.axvline(1e-4)
            # plt.axvline(1e-8)
            plt.grid()
            plt.ylabel('Temperature (T)' )
            plt.xlabel('Density (nH)')
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig(f'{savepath}/{mf}_haloid{haloid}_xraylum-{line}_med_{lower_r}-{upper_r}r200c.png')
            plt.close()
            print('figure has been saved!')


            fig, ax = plt.subplots()
            plt.hexbin(nH[filter], T[filter],  C=None, bins = 'log', xscale = 'log',yscale = 'log', gridsize = 40,  cmap='viridis' )
            # plt.hexbin(nH, T, C = r/r200c, reduce_C_function = np.median, bins = 'log', xscale = 'log',yscale = 'log', gridsize = 40, norm = LogNorm(vmax = 1.5), cmap='viridis' ) # C = r/r200c, 
            # plt.hist2d(nH, T,  bins = [np.logspace(-10,2,100), np.logspace(3,9,100)], norm = LogNorm(),  cmap='viridis', weights = r/r200c) 


            plt.title(f'Halo mass in $10^{{{mf}}}-10^{{{mf+0.5}}} M_{{\odot}}$')
            plt.suptitle(f' for particles in {lower_r}r_{{200c}}- {upper_r} r_{{200c}} ')
            # Add colorbar and labels
            cbar = plt.colorbar()
            # cbar.set_label('log10(Radius/$R_{200c})$')
            # cbar.set_label('cts')
            cbar.set_label(f'counts')
            # plt.axhline(1e5)
            # plt.axhline(1e7)
            # plt.axvline(1e-4)
            # plt.axvline(1e-8)
            plt.grid()
            plt.ylabel('Temperature (T)' )
            plt.xlabel('Density (nH)')
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig(f'{savepath}/{mf}_haloid{haloid}_counts_med_{lower_r}-{upper_r}r200c.png')
            plt.close()
            print('figure has been saved!')