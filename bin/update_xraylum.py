'''
0. Read previous wrong flux file
1. Read particle lum distance
2. multiply wrong flux with the lum distance
3. alter the sum lum as lum in 3d sphere r200c
4. divide part lum with right lum distance: coor_z-(halo_center_z-3.125 cMpc)

'''

'''
1. Select the halo center based on soap cat, select 5 halos for each mass bin
2. Based on halo center select corresponding smallest box in snapshots
3. select only particles in r200c (numba), calculate their lum and save
(4. sum up lum of each halo, make the plot)
abundance to solar: wherea calcium and sulfur are the same as the silicon
['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'calcium', 'sulfur', 'iron']
'''
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import astropy.units as u
from unyt import g, cm, mp
import concurrent.futures
import time
import os



@nb.jit(nopython=True)
def msk_in_r200c(coor, halo_center, r200c):
    # requiret coor shape: [part_num,3]
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        where[i] = (coor[i,0]-halo_center[0])**2 + (coor[i,1]-halo_center[1])**2 +(coor[i,2]-halo_center[2])**2 < (r200c)**2
    return where
     
def alter_lum(wrong_flux, z_coord, redshift, z_halo_center):
    right_lumdist = z_coord-(z_halo_center-3.125)
    lum = wrong_flux * (4*np.pi*(z_coord*(1+redshift))**2)
    right_flux = lum/(4*np.pi*(right_lumdist*(1+redshift))**2)
    return lum, right_flux

def para_alter_lum(sid):
    halo_parts = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halo{sid}_partlum_230404.csv')
    msk = halo_ids==sid
    lum = {}
    flux = {}
    for line in ['o7f', 'o8', 'fe17']:
        lum[line], flux[line] = alter_lum(halo_parts[line], halo_parts['part_zcoords'], 0.1, z_gasmass_center[msk])
    return lum['o7f'], flux['o7f'], lum['o8'], flux['o8'], lum['fe17'], flux['fe17']

start = time.perf_counter()
print(f'Program begin: {start}')
workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot'
mass_filter = np.array([13.0, 13.5, 14.0, 14.5, 15.0])
for mf in mass_filter:
    olddatapath = f'{workpath}/results/results_wrong_wholeboxz_sb/xraysb_csvs_230504_{mf}_groups_1028halos'
    halo_basics = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230404.csv')
    halo_ids = halo_basics['halo_ids']-1
    r200c = halo_basics['r200c']
    x_gasmass_center = halo_basics['x_gasmass_center']
    y_gasmass_center = halo_basics['y_gasmass_center']
    z_gasmass_center = halo_basics['z_gasmass_center']
    gasmass_center = np.array([x_gasmass_center, y_gasmass_center, z_gasmass_center]).T
    savepath = f'{workpath}/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos'
    os.makedirs(savepath, exist_ok = True)
    output = {}
    output['sumlum_o7f'] = np.zeros(r200c.shape); output['sumlum_o8'] = np.zeros(r200c.shape); output['sumlum_fe17'] = np.zeros(r200c.shape) 
    with concurrent.futures.ProcessPoolExecutor(128) as executor:
        for i, result in enumerate(executor.map(para_alter_lum,np.array(halo_ids, dtype = int))):
            halo_parts = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halo{halo_ids[i]}_partlum_230404.csv')
            jointmsk = halo_parts['jointmsk']
            halodoc = {}
            # halodoc['lum_o7f'],halodoc['flux_o7f'], halodoc['lum_o8'], halodoc['flux_o8'], halodoc['lum_fe17'], halodoc['flux_fe17'] = para_alter_lum(halo_ids[i])
            halodoc['lum_o7f'],halodoc['flux_o7f'], halodoc['lum_o8'], halodoc['flux_o8'], halodoc['lum_fe17'], halodoc['flux_fe17'] = result
            # calculate sum lum in r200c
            coor = np.array([halo_parts['part_xcoords'], halo_parts['part_ycoords'], halo_parts['part_zcoords']]).T
            r200cmsk = msk_in_r200c(coor, gasmass_center[i], r200c[i])
            output['sumlum_o7f'][i], output['sumlum_o8'][i], output['sumlum_fe17'][i] = np.nansum(halodoc['lum_o7f'][jointmsk & r200cmsk]), np.nansum(halodoc['lum_o8'][jointmsk & r200cmsk]), np.nansum(halodoc['lum_fe17'][jointmsk & r200cmsk])
            df1 = pd.DataFrame.from_dict(halodoc)
            df1.to_csv(f'{savepath}/xray_linelum_snapshot75_halo{halo_ids[i]}_partlum.csv')  
    df = pd.DataFrame.from_dict(output)
    df.to_csv(f'{savepath}/xray_linelum_inr200c_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv')
    print(f'{savepath}/xray_linelum_inr200c_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv has been saved! ')

finish = time.perf_counter()
print(f'Finished in {(finish-start)/60:.2f} min(s)')