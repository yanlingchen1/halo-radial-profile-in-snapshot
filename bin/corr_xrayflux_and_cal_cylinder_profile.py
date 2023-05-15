import numpy as np
import pandas as pd
from glob import glob

workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot'
mass_filter = np.array([13.0, 13.5, 14.0, 14.5, 15.0])

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
for mf in mass_filter:
    halo_parts = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halo{halo_ids[i]}_partlum_230404.csv')
    jointmsk = halo_parts['jointmsk']
    halodoc = {}
    # halodoc['lum_o7f'],halodoc['flux_o7f'], halodoc['lum_o8'], halodoc['flux_o8'], halodoc['lum_fe17'], halodoc['flux_fe17'] = para_alter_lum(halo_ids[i])
    halodoc['lum_o7f'],halodoc['flux_o7f'], halodoc['lum_o8'], halodoc['flux_o8'], halodoc['lum_fe17'], halodoc['flux_fe17'] = result
    # calculate sum lum in r200c
    coor = np.array([halo_parts['part_xcoords'], halo_parts['part_ycoords'], halo_parts['part_zcoords']]).T
    files = glob(f'{workpath}/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos/xray_linelum_snapshot75_halo*_partlum.csv')
    for file in files:
        dat = pd.read_csv(file)
        for line in ['o7f', 'o8', 'fe17']:
            dat[f'flux_{line}'] = dat[f'lum_{line}']/