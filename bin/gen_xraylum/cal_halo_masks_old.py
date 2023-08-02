'''
1. Select the halo center based on soap cat, select 5 halos for each mass bin
2. Based on halo center select corresponding smallest box in snapshots
3. select only particles in r200c (numba), calculate their lum and save
(4. sum up lum of each halo, make the plot)
abundance to solar: mska calcium and sulfur are the same as the silicon
['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'calcium', 'sulfur', 'iron']
'''

'''
!!!!!!!!!!CAUTION!!!!!!!!!
For data in 230718
The coordinates is last three column! 
corr -> data[y_coor, z_coor, tot_abun_to_solar]
tot_abun_to_solar -> data[x_coor]

in cal_profiles code:
if generate the new data in the future, replace '([olddf_part['part_ycoords'], olddf_part['part_zcoords'], olddf_part['tot_abun_to_solar']])' to ([olddf_part['part_xcoords'], olddf_part['part_ycoords'], olddf_part['part_zcoords']])

'''
### Notice, L1000N3600 starts at 78-reds/0.05, reds = 0, 0.1, 0.2, ...
### While, L1000N1800 has complete cat for every redshift intervaled by 0.05
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import swiftsimio as sw
import pandas as pd
import h5py
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
from interpolate_X_Ray_seperate_lines_cont_with_totsolarabun import interpolate_X_Ray as interp_xray
from interpolate_X_Ray_seperate_lines_cont_with_totsolarabun import interpolate
from unyt import g, cm, mp
import concurrent.futures
import time
import os
from datetime import datetime
from glob import glob

m_nu = [0.02, 0.02, 0.02] * u.eV
DESyr3 = FlatLambdaCDM(H0=68.1, Om0=0.3046, m_nu=m_nu, Ob0=0.0486, Tcmb0=2.725)

import numba as nb
@nb.jit(nopython=True)
def msk_in_cylinder(coor, halo_center, r, z): # r, z in cMpc
    n = 2
    msk = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        msk[i] = (d2 <= r**2) & (coor[i,2] < (halo_center[2]+z/2)) & (coor[i,2] >= (halo_center[2]-z/2))
    return msk  

def interpdens2nH(interpdens, hydrogen_massfrac, redshifts):
    # after np.log10, unyt unit is lost
    scale_factor = 1/(redshifts+1)
    return interpdens.to(g/cm**3) * hydrogen_massfrac / mp.to(g) * (1/scale_factor**3)
 

def cal_halo_summass(sid):
    # load region
    filename = f'/cosma8/data/dp004/flamingo/Runs/{sim}/HYDRO_FIDUCIAL/snapshots/flamingo_00{snapnum}/flamingo_00{snapnum}.hdf5'
    mask = sw.mask(filename)
    boxsize = mask.metadata.boxsize

    load_region = [[(center[:,0][sid]/boxsize[0].value-0.025)*boxsize[0], (center[:,0][sid]/boxsize[0].value+0.025)*boxsize[0]],[(center[:,1][sid]/boxsize[1].value-0.025)*boxsize[1], (center[:,1][sid]/boxsize[1].value+0.025)*boxsize[1]],[(center[:,2][sid]/boxsize[2].value-0.025)*boxsize[2], (center[:,2][sid]/boxsize[2].value+0.025)*boxsize[2]]]
    mask.constrain_spatial(load_region)
    data = sw.load(filename, mask=mask)

    msk = msk_in_cylinder(np.array(data.gas.coordinates), (center[:,0][sid],center[:,1][sid],center[:,2][sid]), r200c[sid]*12, r200c[sid]*24)

    data.gas.redshifts = np.ones(len(data.gas.densities))*reds
    a_agn = data.gas.last_agnfeedback_scale_factors
    data.gas.densities
    data.gas.temperatures

    a_agn = np.array(a_agn)
    t_par = DESyr3.lookback_time(reds)
    t_agn = np.zeros(a_agn.shape)
    t_agn[a_agn!=-1] = np.nan
    t_agn[a_agn!=-1] = DESyr3.lookback_time(1/np.array(a_agn[a_agn!=-1])-1)
    nH_dens = interpdens2nH(data.gas.densities, data.gas.smoothed_element_mass_fractions.hydrogen, np.ones(len(msk)) * reds)

    calxray_msk = (nH_dens.value>1e-8) & (nH_dens.value<1e6) & (data.gas.temperatures.value>1e5) & (data.gas.temperatures.value<np.power(10,9.5))
    exclude_rhp_msk =  ~((abs(t_agn-t_par.value)*1000<15) & (a_agn!=-1) & ((np.power(10, 6.94455)<data.gas.temperatures.value) | (np.power(10, 8.24455)>data.gas.temperatures.value)))

    xray_file = pd.read_csv(f'{datapath}/xray_linelum_snapshot{snapnum}_halo{int(haloids[sid]-1)}_partlum.csv')
    xray_O7_msk = xray_file['o7f']!=0
    xray_O8_msk = xray_file['o8']!=0
    xray_Fe17_msk = xray_file['fe17']!=0


    from IPython import embed
    embed()
    return data.gas.particle_ids, msk, msk & calxray_msk, msk & exclude_rhp_msk, msk & xray_O8_msk, msk & xray_O7_msk, msk & xray_Fe17_msk 


# count time
start = time.perf_counter()

# Caution!!! only redshift 0 soap halo position is correct, for larger redshifts halo postions are wrong!

# define the redshift and the halo mass
reds = 0.1
# sim = 'L1000N1800'
# snapnum = int(77-reds/0.05)

sim = 'L1000N3600'
snapnum = int(78-reds/0.05)

np.random.seed(0)
mass_filter = np.array([13.5])
# halonum = 1028
halonum = 128
for mf in mass_filter:

    path = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/redshift_01/{sim}'
    datapath = f'{path}/xraysb_csvs_230718_{mf}_groups_{halonum}halos_cyl'
    sumfile = pd.read_csv(glob(f'{datapath}/*btw*')[0])

    haloids = sumfile['halo_ids']
    center = np.array([sumfile['x_gasmass_center'], sumfile['y_gasmass_center'], sumfile['z_gasmass_center']]).T
    r200c = sumfile['r200c']

    # preload table
    interp = interpolate()
    print('loading table')
    interp.load_table()
    print('table_loaded')

    # define paths
    workpath = '/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot'
    savepath = f'{datapath}/masks'
    os.makedirs(savepath, exist_ok = True)

    part_ids, radmsk, calxray_msk, ex_rhp_msk, xray_O8_msk, xray_O7_msk, xray_Fe17_msk = cal_halo_summass(0)

    
    # ######## for test ##########
    # halodoc = {}
    # index = int(halo_sel_ids[29]-1)
    # halodoc['o7f'], halodoc['o8'], halodoc['fe17'], halodoc['jointmsk'], halodoc['part_masses'], halodoc['part_dens'], halodoc['nH_dens'], halodoc['part_temperatures'], halodoc['abun_hydrogen'], halodoc['abun_helium'], halodoc['abun_carbon'], halodoc['abun_nitrogen'], halodoc['abun_oxygen'], halodoc['abun_neon'], halodoc['abun_magnesium'], halodoc['abun_silicon'], halodoc['abun_iron'],halodoc['part_xcoords'], halodoc['part_ycoords'], halodoc['part_zcoords'], halodoc['tot_abun_to_solar']  = cal_halo_summass(index)
    # output['o7f'][0], output['o8'][0], output['fe17'][0] = np.nansum(halodoc['o7f'][halodoc['jointmsk']]), np.nansum(halodoc['o8'][halodoc['jointmsk']]), np.nansum(halodoc['fe17'][halodoc['jointmsk']])
    # df1 = pd.DataFrame.from_dict(halodoc)
    # df1.to_csv(f'{savepath}/xray_linelum_snapshot{snapnum}_halo{np.array(halo_sel_ids-1, dtype = int)[0]}_partlum.csv')  
    
    
    # ######### formal ###########
    # with concurrent.futures.ProcessPoolExecutor(8) as executor:
    #     for i, result in enumerate(executor.map(cal_halo_summass, np.arange(len(haloids)))):
    #         halodoc = {}
    #         halodoc['particle_ids'], halodoc['radmsk'], halodoc['calxray_msk'], halodoc['ex_rhp_msk'], halodoc['xray_O8_msk'], halodoc['xray_O7_msk'], halodoc['xray_Fe17_msk'] = result
    #         df1 = pd.DataFrame.from_dict(halodoc)
    #         df1.to_csv(f'{savepath}/masks_snapshot{snapnum}_halo{np.array(haloids-1, dtype = int)[i]}_partlum.csv')  
    #         print(f'{datetime.now()} {haloids[i]}.csv')


finish = time.perf_counter()
print(f'Finished in {(finish-start)/60:.2f} min(s)')