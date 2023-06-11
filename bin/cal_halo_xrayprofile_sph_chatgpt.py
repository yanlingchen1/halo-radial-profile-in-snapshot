import pandas as pd
import numpy as np
import numba as nb
import os
import concurrent.futures
from datetime import datetime

@nb.jit(nopython=True)
def msk_in_sph(coor, halo_center, r1, r2):
    n = 3
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            dist = coor[i, j] - halo_center[j]
            d2 += dist ** 2
        where[i] = (d2 < r2**2) & (d2 >= r1**2)
    return where

def cal_xraylum_excl(args):
    k, prop, halo_cen, xbins, olddf_part = args
    bins = np.power(10, xbins) * 1  # Mpc
    prop_arr = np.zeros(len(bins))
    cts_arr = np.zeros(len(bins))
    for j in range(len(bins) - 1):
        radmsk = msk_in_sph(np.array([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']]).T,
                            halo_cen, bins[j], bins[j + 1])
        if prop != 'cts':
            prop_arr[j] = np.nansum(olddf_part[prop][np.array(olddf_part['jointmsk']) & radmsk])
        else:
            cts_arr[j] = np.nansum(np.array(olddf_part['jointmsk']) & radmsk)
    if prop != 'cts':
        return prop_arr
    else:
        return cts_arr

def cal_xraylum_incl(args):
    k, prop, halo_cen, xbins, olddf_part = args
    bins = np.power(10, xbins) * 1  # Mpc
    prop_arr = np.zeros(len(bins))
    cts_arr = np.zeros(len(bins))
    for j in range(len(bins) - 1):
        radmsk = msk_in_sph(np.array([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']]).T,
                            halo_cen, bins[j], bins[j + 1])
        if prop != 'cts':
            prop_arr[j] = np.nansum(olddf_part[prop][radmsk])
        else:
            cts_arr[j] = np.nansum(radmsk)
    if prop != 'cts':
        return prop_arr
    else:
        return cts_arr

for mf in [13.0, 13.5]:
    print(f'{datetime.now()}: Program begins!')
    olddatapath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_wrong_wholeboxz_sb/xraysb_csvs_230504_{mf}_groups_1028halos'
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_1028halos'
    savepath = f'{workpath}/xraylum_csvs_230610_{mf}_groups_radial_pkpc_cyl'
    os.makedirs(savepath, exist_ok=True)
    df_halo = pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halomass_btw_{int(mf * 10)}_{int((mf + 0.5) * 10)}_230404.csv')
    haloids = df_halo['halo_ids']
    halo_centers = np.array([df_halo['x_gasmass_center'], df_halo['y_gasmass_center'], df_halo['z_gasmass_center']]).T
    halo_r200cs = df_halo['r200c']
    xbins_mean = np.arange(-2, 3.25, 0.25)
    xbins_med = np.arange(-2, 3.1, 0.1)
    props_names = ['part_masses', 'part_dens', 'part_temperatures', 'cts']
    xbins_names = ['025dex', '010dex']

    for q, xbins in enumerate([xbins_mean, xbins_med]):
        print(f'{datetime.now()}:xbins')
        output = {}
        output1 = {}
        for prop in props_names:
            output[prop] = np.zeros((len(xbins), len(haloids)))
            output1[prop] = np.zeros((len(xbins), len(haloids)))

        with concurrent.futures.ProcessPoolExecutor(16) as executor:
            args_excl = [(k, prop, halo_centers[k], xbins, pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halo{int(haloid - 1)}_partlum_230404.csv'))
                         for k, haloid in enumerate(haloids)]
            args_incl = [(k, prop, halo_centers[k], xbins, pd.read_csv(f'{olddatapath}/xray_linelum_snapshot75_halo{int(haloid - 1)}_partlum_230404.csv'))
                         for k, haloid in enumerate(haloids)]

            results_excl = executor.map(cal_xraylum_excl, args_excl)
            results_incl = executor.map(cal_xraylum_incl, args_incl)

            for k, result in enumerate(results_excl):
                output[props_names[k]][:, k] = result

            for k, result in enumerate(results_incl):
                output1[props_names[k]][:, k] = result

        for prop in props_names:
            df = pd.DataFrame.from_dict(output[prop])
            df.to_csv(f'{savepath}/{prop}_{xbins_names[q]}_excl_sph.csv')

            df = pd.DataFrame.from_dict(output1[prop])
            df.to_csv(f'{savepath}/{prop}_{xbins_names[q]}_incl_sph.csv')

        print(f'{datetime.now()}: csv has been saved!')
