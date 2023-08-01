import pandas as pd
import numpy as np
import numba as nb
import os
import concurrent.futures
from datetime import datetime
import matplotlib.pyplot as plt

'''
!!!!!!!!!!CAUTION!!!!!!!!!
For data in 230718
The coordinates is last three column! 
corr -> data[y_coor, z_coor, part_zcoords]
part_zcoords -> data[x_coor]

in cal_profiles code:
if generate the new data in the future, replace '([olddf_part['part_ycoords'], olddf_part['part_zcoords'], olddf_part['part_zcoords']])' to ([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']])

'''

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
        print(np.sum(where))
    return where
   
def msk_in_sph_new(coor, halo_center, r1, r2):
    dists = np.linalg.norm(coor-halo_center, axis=1)
    where = (dists< r2) & (dists>=r1)
    return where

# input parameters
xbins_mean = np.arange(-1.5, 1, 0.25)
xbins_med = np.arange(-1.5, 1, 0.1)
mul_props_names = ['nH_dens', 'part_temperatures', 'abun_oxygen', 'abun_iron']
xbins_names = ['010dex'] # , '025dex'
mul_headers = ['mul_mass', 'mul_vol', 'mul_o7r', 'mul_o8', 'mul_fe17']
props_names = ['part_vol']
mask_names = ['excl']

reds = 0.1
sim = 'L1000N1800'
snapnum = int(77-reds/0.05)

# sim = 'L1000N3600'
# snapnum = int(78-reds/0.05)


# # begin calculate profiles
for mf in [13.5]:
    # set timing
    print(f'{datetime.now()}: Program begins!')

    # set paths
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/redshift_01/{sim}'
    savepath = f'{workpath}/profiles_230718_{mf}_r500c'
    datapath = f'{workpath}/xraysb_csvs_230718_{mf}_groups_1028halos_cyl'
    os.makedirs(savepath, exist_ok = True)

    # read data
    df_halo = pd.read_csv(f'{datapath}/xray_linelum_snapshot{snapnum}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv')
    haloids = df_halo['halo_ids']
    halo_centers = np.array([df_halo['x_gasmass_center'], df_halo['y_gasmass_center'], df_halo['z_gasmass_center']]).T
    halo_r500cs = df_halo['r500c'] * (1+reds)
            
    # calculate multiply profile
    # set timing
    print(f'{datetime.now()}: Program begins to calculate mul profile!')

    # # initialize the results dictionary outside the loop
    # results_by_prop = {prop: {mul_prop: {mask_name: {xbins_name: {} for xbins_name in xbins_names} for mask_name in mask_names} for mul_prop in props_names}  for prop in mul_props_names}

    for q, xbins in enumerate([xbins_med]):
        for prop in mul_props_names:
            for mul_prop in props_names:
                # multiply mass for mass-weighted profile
                # initialize results dictionary outside the loop
                for n, mask_name in enumerate(mask_names):

                    def cal_xraylum_mul(k):
                        # read data
                        haloid = haloids[k]
                        halo_cen = halo_centers[k]
                        # print(f'cal halo{haloid} ...')
                        bins = np.power(10, xbins) * halo_r500cs[k] #Mpc
                        olddf_part = pd.read_csv(f'{datapath}/xray_linelum_snapshot{snapnum}_halo{int(haloid-1)}_partlum.csv') 
                        
                    
                        arr = np.zeros(len(bins)-1)
                        for i in range(len(bins)-1):
                            # load mask
                            radmsk = msk_in_sph_new(np.array([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']]).T, halo_cen, bins[i], bins[i+1])
                            if mask_name == 'excl':
                                mask = np.array(olddf_part['jointmsk']) & radmsk
                            elif mask_name == 'incl':
                                mask = radmsk
                            else:
                                return ValueError('Invalid Mask name!')
                                
                            # calculate data
                            if mul_prop == 'part_vol':
                                data = olddf_part['part_masses'] / olddf_part['part_dens'] * olddf_part[prop]
                            else:
                                data = olddf_part[mul_prop] * olddf_part[prop]
                            arr[i] = np.nansum(data[mask])
                            
                        return arr
                
              

            # # testing
            # res = cal_xraylum_mul(1)
            # print(prop)
            # print(res.keys())
            # print(res.items())
                


                    # formal run
                    output = np.zeros((len(xbins) - 1, len(haloids)))

                    with concurrent.futures.ProcessPoolExecutor(16) as executor:
                        for k, result in enumerate(executor.map(cal_xraylum_mul, np.arange(len(haloids)))): 
                            output[:,k] = result

                    # Save the output files for all properties, masks, and xbins after processing all properties

                    df = pd.DataFrame.from_dict(output)
                    df.to_csv(f'{savepath}/{prop}_mul_{mul_prop}_{xbins_names[q]}_{mask_name}_sph.csv')
                    print(f'{datetime.now()}: {prop}_mul_{mul_prop}_{xbins_names[q]}_{mask_name}_sph.csv has been saved!')

