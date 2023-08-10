import pandas as pd
import numpy as np
import numba as nb
import os
import concurrent.futures
from datetime import datetime
import matplotlib.pyplot as plt
from glob import glob
import h5py

'''
!!!!!!!!!!CAUTION!!!!!!!!!
For data in 230718
The coordinates is last three column! 
corr -> data[y_coor, z_coor, part_zcoords]
part_zcoords -> data[x_coor]

if generate the new data in the furture, replace '([olddf_part['part_ycoords'], olddf_part['part_zcoords'], olddf_part['part_zcoords']])' to ([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']])
'''


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


# set bins
xbins_mean = np.arange(-1.5, 1, 0.25)
# xbins_med = np.arange(-1.5, 1, 0.1)
xbins_med = np.log10(np.logspace(-2, np.log10(3), 30))
# set saved varnames
props_names = ['part_masses']#,'part_masses','part_vol', 'cts', 'o7f', 'o8', 'fe17' 'tot_abun', 'cts', 'o7f', 'o8', 'fe17', 'part_vol']
xbins_names = ['010dex'] # 025dex
mask_names = ['jointmsk']

for mf in [13.0, 13.5, 14.0, 14.5, 15.0]: # 13.0, 13.5, 14.0, 14.5, 15.0
    print(f'{datetime.now()}: Program begins!')

    # input parameters
    reds = 0.1
    sim = 'L1000N1800'
    snapnum = int(77-reds/0.05)
    halonum = 32

    # sim = 'L1000N3600'
    # snapnum = int(78-reds/0.05)
    # halonum = 128

    # set paths
    workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/redshift_01/{sim}'
    savepath = f'{workpath}/profiles_230809_{mf}_ind_r500c_FINEBIN'
    datapath = f'{workpath}/xraysb_csvs_230809_{mf}_groups_{halonum}halos_sph'
    os.makedirs(savepath, exist_ok = True)
    
    # load halo summary file
    df_halo = pd.read_csv(f'{datapath}/xray_linelum_snapshot{snapnum}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv')

    haloids = df_halo['halo_ids']
    # haloids = np.array(df_halo['halo_ids'] - 1).astype(int)
    halo_centers = np.array([df_halo['x_gasmass_center'], df_halo['y_gasmass_center'], df_halo['z_gasmass_center']]).T

    # load r500c in soap
    with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/{sim}/HYDRO_FIDUCIAL/SOAP/halo_properties_00{snapnum}.hdf5", 'r') as catalogue_soap:
        r500c_sp = catalogue_soap["SO/500_crit/SORadius"][()] * (1+reds)

    halo_r500cs = r500c_sp[np.array(haloids-1).astype(int)]

    for q, xbins in enumerate([xbins_med]): # , xbins_med
        for prop in props_names:
            for mask_name in mask_names:
                def cal_xraylum_excl(k):
                    # load parameter of single halo 
                    haloid = haloids[k]
                    halo_cen = halo_centers[k]

                    bins = np.power(10, xbins) * halo_r500cs[k] #Mpc
                    olddf_part = pd.read_csv(f'{datapath}/xray_linelum_snapshot{snapnum}_halo{int(haloid-1)}_partlum.csv')

                    # initialize
                    arr  = np.zeros(len(bins))
                    for j in range(len(bins)-1):
                        #part_zcoords -> part_zcoords
                        radmsk = msk_in_sph_new(np.array([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']]).T, halo_cen, bins[j], bins[j+1])

                        mask = olddf_part['jointmsk']
                        # from IPython import embed
                        # embed()
                        if prop == 'cts':
                            arr[j] = np.nansum(mask & radmsk)
                        elif prop == 'part_vol':
                            arr[j] = np.nansum((olddf_part['part_masses'][mask & radmsk]/olddf_part['part_dens'])[mask & radmsk])
                        else:
                            arr[j] = np.nansum(olddf_part[prop][mask & radmsk])

                    return arr
                    

                # # testing
                # for i in range(1,2):
                #     print(prop)
                #     print(cal_xraylum_excl(i))


                ## formal
                with concurrent.futures.ProcessPoolExecutor(32) as executor:
                    output = np.zeros((len(xbins), len(haloids)))

                    for k, result in enumerate(executor.map(cal_xraylum_excl, np.arange(32))): #len(haloids)
                        print(f'{datetime.now()}:{k}')
                        output[:,k] = result

                    df = pd.DataFrame.from_dict(output)
                    df.to_csv(f'{savepath}/{prop}_{xbins_names[q]}_{mask_name}_sph.csv')
                    print(f'{datetime.now()}: csv has been saved!')


               