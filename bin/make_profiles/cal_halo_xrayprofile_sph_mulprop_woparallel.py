import pandas as pd
import numpy as np
import numba as nb
import os
import concurrent.futures
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm


# define functions
@nb.jit(nopython=True)
def msk_in_sph(coor, halo_center, r1, r2):
    n = 3
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

# input parameters
xbins_mean = np.arange(-2,3.25,0.25)
xbins_med = np.arange(-2,3.1,0.1)
mul_props_names = ['part_temperatures', 'nH_dens', 'abun_oxygen', 'abun_iron'] # , 'abun_hydrogen', 'abun_oxygen', 'abun_iron'
xbins_names = ['025dex', '010dex']
mul_headers = ['mul_mass', 'mul_vol', 'mul_o7f', 'mul_o8', 'mul_fe17']
mask_names = ['excl', 'incl']

reds = 0.1
# sim = 'L1000N1800'
# snapnum = int(77-reds/0.05)

sim = 'L1000N3600'
snapnum = int(78-reds/0.05)

# # begin calculate profiles
for mf in [13.5]:
    # set timing
    print(f'{datetime.now()}: Program begins!')

    # set paths
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/redshift_01/{sim}'
    savepath = f'{workpath}/profiles_230724_{mf}_nopara'
    datapath = f'{workpath}/xraysb_csvs_230718_{mf}_groups_1028halos_cyl'
    os.makedirs(savepath, exist_ok = True)

    # read data
    df_halo = pd.read_csv(f'{datapath}/xray_linelum_snapshot{snapnum}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv')
    haloids = df_halo['halo_ids']
    halo_centers = np.array([df_halo['x_gasmass_center'], df_halo['y_gasmass_center'], df_halo['z_gasmass_center']]).T
    halo_r200cs = df_halo['r200c']
            
    # calculate multiply profile
    # set timing
    print(f'{datetime.now()}: Program begins to calculate mul profile!')
    for q, xbins in enumerate([xbins_med]): # , xbins_med
        for prop in mul_props_names:
            # multiply mass for mass-weighted profile
            # initialize output
            output = {}
            for mask_name in mask_names:
                for data_name in mul_headers:
                    output[f'{data_name}_{mask_name}'] = np.zeros((len(xbins), len(haloids)))
            for k in tqdm(range(len(haloids))): # 
                # read data
                haloid = haloids[k]
                halo_cen = halo_centers[k]
                # print(f'cal halo{haloid} ...')
                bins = np.power(10, xbins) * 1 #Mpc
                olddf_part = pd.read_csv(f'{datapath}/xray_linelum_snapshot{snapnum}_halo{int(haloid-1)}_partlum.csv') 
                

                datas = [olddf_part['part_masses'] * olddf_part[prop], 
                            olddf_part['part_masses'] / olddf_part['part_dens'] * olddf_part[prop],
                            olddf_part['o7f'] * olddf_part[prop],
                            olddf_part['o8'] * olddf_part[prop],
                            olddf_part['fe17'] * olddf_part[prop]]
                for i, data in enumerate(datas):
                    data_name = mul_headers[i]
                    for j in range(len(bins)-1):
                        radmsk = msk_in_sph(np.array([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']]).T, halo_cen, bins[j], bins[j+1])
                        output[f'{data_name}_excl'][j, k] = np.nansum(data[np.array(olddf_part['jointmsk']) & radmsk])
                        output[f'{data_name}_incl'][j, k] = np.nansum(data[radmsk])
            # # testing
            # res = output
            # print(prop)
            # print(res.keys())
            # print(res.items())
            
            
            # save 2d array to csv, don't save 3d dictionaries!
            for mask_name in mask_names:
                for data_name in mul_headers:
                    print(f'{data_name}_{mask_name}')
                    print(output[f'{data_name}_{mask_name}'])
                    df = pd.DataFrame.from_dict(output[f'{data_name}_{mask_name}'])  # Specify the index explicitly
                    df.to_csv(f'{savepath}/{prop}_{xbins_names[q]}_{data_name}_{mask_name}_sph.csv')
                    print(f'{datetime.now()}: csv has been saved!')