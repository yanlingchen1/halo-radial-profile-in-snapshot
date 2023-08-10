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
corr -> data[y_coor, z_coor, tot_abun_to_solar]
tot_abun_to_solar -> data[x_coor]

in cal_profiles code:
if generate the new data in the future, replace '([olddf_part['part_ycoords'], olddf_part['part_zcoords'], olddf_part['tot_abun_to_solar']])' to ([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']])

'''


@nb.jit(nopython=True)
def msk_in_cylinder(coor, halo_center, r, z): # r, z in cMpc
    n = 2
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = (d2 <= r**2) & (coor[i,2] < (halo_center[2]+z/2)) & (coor[i,2] >= (halo_center[2]-z/2))
    return where  

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

def same_row_in_2darr(a, b):
    return (a[:, None] == b).all(-1).any(-1)


# input parameters
# xbins_mean = np.arange(-1.5, 1, 0.25)
# xbins_med = np.arange(-1.5, 1, 0.1)
xbins_med = np.arange(-1.5, 0.5, 0.1)
xbins_mean = np.arange(-1.5, 0.5, 0.25)
mul_props_names = ['part_temperatures', 'abun_iron', 'nH_dens'] # 'part_masses', 
xbins_names = ['010dex']
props_names = ['part_vol'] # 'part_masses', 'part_vol', 'o7f', 'o8', 'fe17'
mask_names = [ 'rhp'] #, 'incl'

reds = 0.1
sim = 'L1000N1800'
snapnum = int(77-reds/0.05)

# sim = 'L1000N3600'
# snapnum = int(78-reds/0.05)

halonum = 1028

# # begin calculate profiles
for mf in [13.5]:
    # set timing
    print(f'{datetime.now()}: Program begins!')

    # set paths
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/redshift_01/{sim}'
    savepath = f'{workpath}/profiles_230718_{mf}_paratest_abun'
    datapath = f'{workpath}/xraysb_csvs_230718_{mf}_groups_{halonum}halos_cyl'
    os.makedirs(savepath, exist_ok = True)

    # read data
    df_halo = pd.read_csv(f'{datapath}/xray_linelum_snapshot{snapnum}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv')
    haloids = df_halo['halo_ids']
    halo_centers = np.array([df_halo['x_gasmass_center'], df_halo['y_gasmass_center'], df_halo['z_gasmass_center']]).T
    halo_r200cs = df_halo['r200c'] * (1+reds)
            
    # calculate multiply profile
    # set timing
    print(f'{datetime.now()}: Program begins to calculate mul profile!')

    for q, xbins in enumerate([xbins_med]):
        for prop in mul_props_names:
            # multiply mass for mass-weighted profile
            #initialize
            for n, mask_name in enumerate(mask_names):
                for mul_prop in props_names:

                    def cal_xraylum_mul(k):
                        # read data
                        haloid = haloids[k]
                        halo_cen = halo_centers[k]
                        # print(f'cal halo{haloid} ...')
                        bins = np.power(10, xbins) * halo_r200cs[k] #Mpc


                        # 
                        olddf_part = pd.read_csv(f'{datapath}/xray_linelum_snapshot{snapnum}_halo{int(haloid-1)}_partlum.csv') 
                        data_part = pd.read_csv(f'{datapath}/tot_abun_snapshot{snapnum}_halo{int(haloid-1)}.csv')
                        file_partid = np.load(f'{datapath}/part_ids/msk_ids_snapshot{snapnum}_halo{int(haloid-1)}_{sim}_{halonum}halos.npy')
                        
                        old_coords = np.array([olddf_part['part_ycoords'], olddf_part['part_zcoords'], olddf_part['tot_abun_to_solar']]).T
                        new_coords = np.array([data_part['part_xcoords'], data_part['part_ycoords'], data_part['part_zcoords']]).T
                        '''
                        The olddf_part save all the particles in the smallest snapshot box (~30Mpc) (Should be corrected later)
                        while the data_part only save all the particles in the cylinder.
                        Since the particle id is not saved (need to be corrected later)
                        We need to correspond the particles via the coordinates. 
                        So if coor1 ==coor2, we take the particles for calculating.
                        
                        '''

                        coords_msk = file_partid
                        # coords_msk = np.isin(old_coords[:,0], new_coords[:,0]) & np.isin(old_coords[:,1], new_coords[:,1]) & np.isin(old_coords[:,2], new_coords[:,2])

                        arr = np.zeros(len(bins))
                        # 
                        for i in range(len(bins)-1):
                            # load mask
                            new_radmsk = msk_in_sph_new(new_coords, halo_cen, bins[i], bins[i+1])
                            old_radmsk = msk_in_sph_new(old_coords, halo_cen, bins[i], bins[i+1])
                            # load mask
                            maskfile = pd.read_csv(f'{datapath}/masks/masks_snapshot{snapnum}_halo{int(haloids[k]-1)}_partlum.csv')

                            if mask_name == 'total':
                                old_msk = np.array(maskfile['radmsk']) & old_radmsk
                                new_msk = np.array(maskfile['radmsk'])[coords_msk] & new_radmsk
                            elif mask_name == 'rhp':
                                old_msk = np.array(maskfile['ex_rhp_msk']) & old_radmsk
                                new_msk = np.array(maskfile['ex_rhp_msk'])[coords_msk] & new_radmsk
                            elif mask_name == 'rhp_xraylum':
                                old_msk = np.array(maskfile['xray_O8_msk'] & maskfile['ex_rhp_msk']) & old_radmsk
                                new_msk = np.array(maskfile['xray_O8_msk'] & maskfile['ex_rhp_msk'])[coords_msk] & new_radmsk


                            # from IPython import embed
                            # embed()
                            # calculate data
                            if mul_prop == 'part_vol':
                                data = np.array(olddf_part['part_masses'][old_msk]) / np.array(olddf_part['part_dens'][old_msk]) * np.array(data_part[prop][new_msk])
                            else:
                                data = np.array(olddf_part[mul_prop][old_msk]) * np.array(data_part[prop][new_msk])

                            arr[i] = np.nansum(data)
                        return arr

                    # # # testing
                    res = cal_xraylum_mul(18)
                    print(res)
                    # # #     # try:
                    # # #     #     res = cal_xraylum_mul(1) 
                    # # #     # except:
                    # # #     #     print(i)                   

                
                    # # formal run
                    # output = np.zeros((len(xbins), len(haloids)))

                    # with concurrent.futures.ProcessPoolExecutor(32) as executor:
                    #     for k, result in enumerate(executor.map(cal_xraylum_mul, np.arange(32))): #
                    #         # print(k)
                    #         output[:,k] = result

                    # # Save the output files for all properties, masks, and xbins after processing all properties
                    # df = pd.DataFrame.from_dict(output)
                    # df.to_csv(f'{savepath}/{prop}_mul_{mul_prop}_{xbins_names[q]}_{mask_name}_sph.csv')
                    # print(f'{datetime.now()}: {prop}_mul_{mul_prop}_{xbins_names[q]}_{mask_name}_sph.csv has been saved!')

