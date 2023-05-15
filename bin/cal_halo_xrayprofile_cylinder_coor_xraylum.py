import pandas as pd
import numpy as np
import numba as nb
import os
import concurrent.futures


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
   

for mf in [13.0, 13.5, 14.0, 14.5, 15.0]:
    olddatapath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_wrong_wholeboxz_sb/xraysb_csvs_230504_{mf}_groups_1028halos'
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos'
    savepath = f'{workpath}/xraylum_csvs_230511_{mf}_groups_radial_pkpc_cylinder'
    os.makedirs(savepath, exist_ok = True)
    df_halo = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230404.csv')
    haloids = df_halo['halo_ids']
    halo_centers = np.array([df_halo['x_gasmass_center'], df_halo['y_gasmass_center'], df_halo['z_gasmass_center']]).T
    halo_r200cs = df_halo['r200c']
    xbins = np.linspace(-2,3.1,50)
    props = [('o7f', 'sum'), ('o8', 'sum'), ('fe17', 'sum')]


    for prop in props:
        output = np.zeros((len(xbins), len(haloids)))
        with concurrent.futures.ProcessPoolExecutor(128) as executor:
            for k, result in enumerate(executor.map(cal_xraylum,np.arange(len(haloids)))):
                output[:,k] = result
        # # for test
        # k = 0
        # output[:,k] = cal_xraylum(k)
        df = pd.DataFrame.from_dict(output)
        df.to_csv(f'{savepath}/{prop[0]}.csv')