'''
1. read part lum for every halo in new files
2. include recent heated particles (not apply jointmsk)
3. sum only parts in r200c of every halo
'''

import pandas as pd
import numpy as np
import numba as nb
import os
import concurrent.futures

@nb.jit(nopython=True)
def msk_in_radii_cyl(coor, halo_center, r1, r2, z):
    n = 2
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = (d2 < r2**2) & (d2 >= r1**2) & (coor[i,2] <= (halo_center[2]+z)) & (coor[i,2] >= (halo_center[2]-z))
    return where

@nb.jit(nopython=True)
def msk_in_r_cyl(coor, halo_center, r, z):
    n = coor.shape[1]
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(2):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = (d2 <= r**2) & (coor[i,2] <= (halo_center[2]+z/2)) & (coor[i,2] >= (halo_center[2]-z/2))
    return where

@nb.jit(nopython=True)
def msk_in_r_sph(coor, halo_center, r):
    n = coor.shape[1]
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = d2 <= r**2
    return where

@nb.jit(nopython=True)
def msk_in_cylinder(coor, halo_center, r1,r2, z): # r, z in cMpc
    n = 2
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = (d2 <= r2**2) & (d2 >= r1**2) & (coor[i,2] < (halo_center[2]+z/2)) & (coor[i,2] >= (halo_center[2]-z/2))
    return where

# right_lumdist = 4*np.pi*((1.48e27/3.0857e24)**2)
xbins = np.linspace(-2,3.1,50)
bins = np.power(10, xbins) * 1 #Mpc

for mf in [13.0, 13.5, 14.0, 14.5, 15.0]:
    olddatapath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_wrong_wholeboxz_sb/xraysb_csvs_230504_{mf}_groups_1028halos'
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos'
    savepath = f'{workpath}/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos'
    os.makedirs(savepath, exist_ok = True)
    df_halo = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230404.csv')
    haloids = df_halo['halo_ids']
    halo_centers = np.array([df_halo['x_gasmass_center'], df_halo['y_gasmass_center'], df_halo['z_gasmass_center']]).T
    r200c = df_halo['r200c']
    xbins = np.linspace(-2,3.1,50)


    output = {}
    for line_type in ['o7f', 'o8', 'fe17']:
        for profile_type in ['sph', 'cyl']:
            for recent_heat_part_type in ['excl', 'incl']:
                output[f'sumlum_{line_type}_{profile_type}_{recent_heat_part_type}'] = np.zeros(r200c.shape)

    for i in range(len(haloids)):
        halo_parts = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halo{haloids[i]-1}_partlum_230404.csv')
        halodoc = pd.read_csv(f'{workpath}/xray_linelum_snapshot75_halo{haloids[i]-1}_partlum.csv')
        jointmsk = halo_parts['jointmsk']
        
        coor = np.array([halo_parts['part_xcoords'], halo_parts['part_ycoords'], halo_parts['part_zcoords']]).T
        # calculate sum lum in r200c
        r200cmsk = msk_in_r_sph(coor, halo_centers[i], r200c[i])
        # print(np.nansum(halodoc['lum_o7f'][jointmsk & r200cmsk]), np.nansum(halodoc['lum_o8'][jointmsk & r200cmsk]), np.nansum(halodoc['lum_fe17'][jointmsk & r200cmsk]))
        # print(np.nansum(halodoc['lum_o7f'][r200cmsk]), np.nansum(halodoc['lum_o8'][r200cmsk]), np.nansum(halodoc['lum_fe17'][r200cmsk]))
        output['sumlum_o7f_sph_excl'][i], output['sumlum_o8_sph_excl'][i], output['sumlum_fe17_sph_excl'][i] = np.nansum(halodoc['lum_o7f'][jointmsk & r200cmsk]), np.nansum(halodoc['lum_o8'][jointmsk & r200cmsk]), np.nansum(halodoc['lum_fe17'][jointmsk & r200cmsk])
        output['sumlum_o7f_sph_incl'][i], output['sumlum_o8_sph_incl'][i], output['sumlum_fe17_sph_incl'][i] = np.nansum(halodoc['lum_o7f'][r200cmsk]), np.nansum(halodoc['lum_o8'][r200cmsk]), np.nansum(halodoc['lum_fe17'][r200cmsk])
        r200cmsk = msk_in_r_cyl(coor, halo_centers[i], r200c[i], 6.25)
        # print( np.nansum(halodoc['lum_o7f'][jointmsk & r200cmsk]), np.nansum(halodoc['lum_o8'][jointmsk & r200cmsk]), np.nansum(halodoc['lum_fe17'][jointmsk & r200cmsk]))
        # print(np.nansum(halodoc['lum_o7f'][r200cmsk]), np.nansum(halodoc['lum_o8'][r200cmsk]), np.nansum(halodoc['lum_fe17'][r200cmsk]))
        output['sumlum_o7f_cyl_excl'][i], output['sumlum_o8_cyl_excl'][i], output['sumlum_fe17_cyl_excl'][i] = np.nansum(halodoc['lum_o7f'][jointmsk & r200cmsk]), np.nansum(halodoc['lum_o8'][jointmsk & r200cmsk]), np.nansum(halodoc['lum_fe17'][jointmsk & r200cmsk])
        output['sumlum_o7f_cyl_incl'][i], output['sumlum_o8_cyl_incl'][i], output['sumlum_fe17_cyl_incl'][i] = np.nansum(halodoc['lum_o7f'][r200cmsk]), np.nansum(halodoc['lum_o8'][r200cmsk]), np.nansum(halodoc['lum_fe17'][r200cmsk])

    df = pd.DataFrame.from_dict(output)
    df.to_csv(f'{savepath}/xray_linelum_inr200c_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230515.csv')
    print(f'{savepath}/xray_linelum_inr200c_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230515.csv has been saved! ')
