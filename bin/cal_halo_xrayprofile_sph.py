import pandas as pd
import numpy as np
import numba as nb
import os
import concurrent.futures
from datetime import datetime
import matplotlib.pyplot as plt


@nb.jit(nopython=True)
def msk_in_sph(coor, halo_center, r1, r2):
    n = 3
    print(coor.shape)
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for k in range(n):
            d2 += (coor[i,k] - halo_center[k])**2
        where[i] = (d2 < r2**2) & (d2 >= r1**2)
    
    return where
   
def msk_in_sph_new(coor, halo_center, r1, r2):
    dists = np.linalg.norm(coor-halo_center, axis=1)
    where = (dists< r2) & (dists>=r1)
    return where

for mf in [14.0, 14.5]:
    print(f'{datetime.now()}: Program begins!')
    olddatapath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_wrong_wholeboxz_sb/xraysb_csvs_230504_{mf}_groups_1028halos'
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos'
    savepath = f'{workpath}/xraylum_csvs_230612_{mf}_groups_radial_pkpc_cyl'
    os.makedirs(savepath, exist_ok = True)
    df_halo = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230404.csv')
    haloids = df_halo['halo_ids']
    halo_centers = np.array([df_halo['x_gasmass_center'], df_halo['y_gasmass_center'], df_halo['z_gasmass_center']]).T
    halo_r200cs = df_halo['r200c']
    xbins_mean = np.arange(-2,3.25,0.25)
    xbins_med = np.arange(-2,3.1,0.1)
    props_names = ['part_masses','part_dens','part_temperatures', 'cts', 'o7f', 'o8', 'fe17']
    xbins_names = ['025dex', '010dex']
    for q, xbins in enumerate([xbins_mean, xbins_med]):
        for prop in props_names:
            def cal_xraylum_excl(k):
                haloid = haloids[k]
                halo_cen = halo_centers[k]
                # print(f'cal halo{haloid} ...')
                bins = np.power(10, xbins) * 1 #Mpc
                olddf_part = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halo{int(haloid-1)}_partlum_230404.csv')
                arr  = np.zeros(len(bins))
                for j in range(len(bins)-1):
                    radmsk = msk_in_sph_new(np.array([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']]).T, halo_cen, bins[j], bins[j+1])
                    if prop != 'cts':
                        arr[j] = np.nansum(olddf_part[prop][np.array(olddf_part['jointmsk']) & radmsk])
                    else:
                        arr[j] = np.nansum(np.array(olddf_part['jointmsk']) & radmsk)
                return arr
           
            def cal_xraylum_incl(k):
                haloid = haloids[k]
                halo_cen = halo_centers[k]
                # print(f'cal halo{haloid} ...')
                bins = np.power(10, xbins) * 1 #Mpc
                arr = np.zeros(len(bins))
                olddf_part = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halo{int(haloid-1)}_partlum_230404.csv')
                for j in range(len(bins)-1):
                    radmsk = msk_in_sph_new(np.array([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']]).T, halo_cen, bins[j], bins[j+1])
                    if prop != 'cts':
                        arr[j] = np.nansum(olddf_part[prop][radmsk])
                    else:
                        arr[j] = np.nansum(radmsk)
                return arr
            ## testing
            # for i in range(1,2):
                # print(cal_xraylum_excl(i))
                # print(cal_xraylum_incl(i))

            output = np.zeros((len(xbins), len(haloids)))
            output1 = np.zeros((len(xbins), len(haloids)))
            with concurrent.futures.ProcessPoolExecutor(64) as executor:
                for k, result in enumerate(executor.map(cal_xraylum_excl, np.arange(len(haloids)))):
                    print(f'{datetime.now()}:{k}')
                    output[:,k] = result
                for k, result in enumerate(executor.map(cal_xraylum_incl, np.arange(len(haloids)))):
                    output1[:,k] = result
            
            df = pd.DataFrame.from_dict(output)
            df.to_csv(f'{savepath}/{prop}_{xbins_names[q]}_excl_sph.csv')
            df = pd.DataFrame.from_dict(output1)
            df.to_csv(f'{savepath}/{prop}_{xbins_names[q]}_incl_sph.csv')
            print(f'{datetime.now()}: csv has been saved!')