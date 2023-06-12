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



# define parameters
reds = 0
mf = 13.0
halonum = 1 # first use 50 halos to test


# define paths
resultpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results'
newdatapath = f'{resultpath}/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos'
savepath = f'./'
olddatapath = f'{resultpath}/results_wrong_wholeboxz_sb/xraysb_csvs_230504_{mf}_groups_1028halos'

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
    coor = np.array([part_halofile['part_xcoords'], part_halofile['part_ycoords'], part_halofile['part_zcoords']]).T
    r = np.linalg.norm(coor - halo_center, axis=1)  # Radius values
    # read densities and temperatures of the particles
    nH = part_halofile['nH_dens']  # Density values
    T = part_halofile['part_temperatures']  # Temperature values
    # Create a scatter plot with color coding
    # plt.hexbin(nH, T, C=r/r200c, cmap='coolwarm', gridsize=200)
    plt.scatter(nH, T, c=r/r200c, cmap='coolwarm', alpha = 0.2, s=1)

    
plt.title(f'Particle Profiles with Temperature Markings within {halonum} halos')
plt.suptitle('Halo mass in $10^{13}-10^{13.5} M_{\odot}$ ')
# Add colorbar and labels
cbar = plt.colorbar()
cbar.set_label('Radius/$R_{200c}$')
plt.axhline(1e5)
plt.axhline(1e7)
plt.axvline(1e-4)
plt.axvline(1e-8)
plt.grid()
plt.ylabel('Temperature (T)' )
plt.xlabel('Density (nH)')
plt.xscale('log')
plt.yscale('log')
plt.savefig('tst.png')
print('figure has been saved!')