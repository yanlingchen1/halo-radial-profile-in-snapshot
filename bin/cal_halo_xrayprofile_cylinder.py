import pandas as pd
import numpy as np
import numba as nb
import os
import concurrent.futures

@nb.jit(nopython=True)
def msk_in_r(coor, halo_center, r):
    n = coor.shape[1]
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = d2 < r**2
    return where
@nb.jit(nopython=True)
def msk_in_radii(coor, halo_center, r1, r2):
    n = coor.shape[1]
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = (d2 < r2**2) & (d2 >= r1**2)
    return where

@nb.jit(nopython=True)
def msk_in_cylinder(coor, halo_center, r1, r2, z):
    n = 2
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = (d2 < r2**2) & (d2 >= r1**2) & (coor[i,2] < (halo_center[2]+z)) & (coor[i,2] >= (halo_center[2]-z))
    return where

for mf in [13.5, 14.0, 14.5]:
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/gen_xray_pipeline/230318/cal_halo_lum_by_halo_230331/xraylum_csvs_230419_{mf}_groups_128halos'
    os.makedirs(f'{workpath}/png', exist_ok=True)
    df_halo = pd.read_csv(f'{workpath}/xray_linelum_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230404.csv')
    haloids = df_halo['halo_ids']
    halo_centers = np.array([df_halo['x_gasmass_center'], df_halo['y_gasmass_center'], df_halo['z_gasmass_center']]).T
    halo_r200cs = df_halo['r200c']
    xbins = np.linspace(-2,3.1,50)
    props = [('o7f', 'sum'), ('o8', 'sum'), ('fe17', 'sum')]
    def cal_xraylum(k):
        haloid = haloids[k]
        halo_cen = halo_centers[k]
        r200c = halo_r200cs[k]
        # print(f'cal halo{haloid} ...')
        bins = np.power(10, xbins) * 1 #Mpc
        prop_arr  = np.zeros(len(bins))
        df_part = pd.read_csv(f'{workpath}/xray_linelum_snapshot75_halo{int(haloid-1)}_partlum_230404.csv')
        for j in range(len(bins)-1):
            radmsk = msk_in_cylinder(np.array([df_part['part_xcoords'], df_part['part_ycoords'], df_part['part_zcoords']]).T, halo_cen, bins[j], bins[j+1], 3)
            prop_arr[j] = np.nansum(df_part[prop[0]][np.array(df_part['jointmsk']) & radmsk])
        return prop_arr

    output = {}
    for prop in props:
        output[prop[0]] = np.zeros((len(xbins), len(haloids)))
        with concurrent.futures.ProcessPoolExecutor(32) as executor:
            for k, result in enumerate(executor.map(cal_xraylum,np.arange(len(haloids)))):
                output[prop[0]][:,k] = result
        savepath = f'{workpath}/xraylum_csvs_230419_{mf}_groups_radial_pkpc_cylinder_new'
        os.makedirs(savepath, exist_ok = True)
        df = pd.DataFrame.from_dict(output[prop[0]])
        df.to_csv(f'{savepath}/{prop[0]}.csv')