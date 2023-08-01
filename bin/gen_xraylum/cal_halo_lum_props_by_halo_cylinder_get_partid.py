'''
1. Select the halo center based on soap cat, select 5 halos for each mass bin
2. Based on halo center select corresponding smallest box in snapshots
3. select only particles in r200c (numba), calculate their lum and save
(4. sum up lum of each halo, make the plot)
abundance to solar: wherea calcium and sulfur are the same as the silicon
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

m_nu = [0.02, 0.02, 0.02] * u.eV
DESyr3 = FlatLambdaCDM(H0=68.1, Om0=0.3046, m_nu=m_nu, Ob0=0.0486, Tcmb0=2.725)

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

def interpdens2nH(interpdens, hydrogen_massfrac, redshifts):
    # after np.log10, unyt unit is lost
    scale_factor = 1/(redshifts+1)
    return interpdens.to(g/cm**3) * hydrogen_massfrac / mp.to(g) * (1/scale_factor**3)


class gasfrac:
    def init(self):
        pass
    def mask(self, data, where):
        self.hydrogen = data.gas.smoothed_element_mass_fractions.hydrogen[()][where]
        self.helium = data.gas.smoothed_element_mass_fractions.helium[()][where]
        self.carbon = data.gas.smoothed_element_mass_fractions.carbon[()][where]
        self.nitrogen = data.gas.smoothed_element_mass_fractions.nitrogen[()][where]
        self.oxygen = data.gas.smoothed_element_mass_fractions.oxygen[()][where]
        self.neon = data.gas.smoothed_element_mass_fractions.neon[()][where]
        self.magnesium = data.gas.smoothed_element_mass_fractions.magnesium[()][where]
        self.silicon = data.gas.smoothed_element_mass_fractions.silicon[()][where]
        self.iron = data.gas.smoothed_element_mass_fractions.iron[()][where]



def cal_halo_summass(sid):
    # load region
    filename = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_00{snapnum}/flamingo_00{snapnum}.hdf5'
    mask = sw.mask(filename)
    boxsize = mask.metadata.boxsize
    load_region = [[(gasmass_center[:,0][sid]/boxsize[0].value-0.025)*boxsize[0], (gasmass_center[:,0][sid]/boxsize[0].value+0.025)*boxsize[0]],[(gasmass_center[:,1][sid]/boxsize[1].value-0.025)*boxsize[1], (gasmass_center[:,1][sid]/boxsize[1].value+0.025)*boxsize[1]],[(gasmass_center[:,2][sid]/boxsize[2].value-0.025)*boxsize[2], (gasmass_center[:,2][sid]/boxsize[2].value+0.025)*boxsize[2]]]
    mask.constrain_spatial(load_region)
    data = sw.load(filename, mask=mask)

    return data.gas.particle_ids

# count time
start = time.perf_counter()

# Caution!!! only redshift 0 soap halo position is correct, for larger redshifts halo postions are wrong!

# define the redshift and the halo mass
reds = 0.1
sim = 'L1000N1800'
snapnum = int(77-reds/0.05)

# sim = 'L1000N3600'
# snapnum = int(78-reds/0.05)

np.random.seed(0)
mass_filter = np.array([13.5])
halonum = 128


# load soap cat
print('loading soap cat...')
with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_00{snapnum}.hdf5", 'r') as catalogue_soap:
    soap_ids = np.array(catalogue_soap["VR/ID"][()])
    m200c_sp = catalogue_soap["SO/200_crit/TotalMass"][()]
    r200c_sp = catalogue_soap["SO/200_crit/SORadius"][()] * (1+reds)
    gasmass_center = catalogue_soap["SO/200_crit/GasCentreOfMass"][()] * (1+reds)
    # mass_center = catalogue_soap["SO/200_crit/CentreOfMass"][()] # mass center and gasmass center are similar
    # BH_center = catalogue_soap["SO/200_crit/MostMassiveBlackHolePosition"][()]  # not right
    xray_bol_emilum = catalogue_soap["SO/200_crit/XRayLuminosityWithoutRecentAGNHeating"][()]
    xray_bol_phlum = catalogue_soap["SO/200_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][()]

# preload table
interp = interpolate()
print('loading table')
interp.load_table()
print('table_loaded')


for mf in mass_filter:

    # define paths
    workpath = '/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot'
    savepath = f'{workpath}/results/redshift_01/L1000N1800/xraysb_csvs_230718_{mf}_groups_{halonum}halos_cyl'
    os.makedirs(savepath, exist_ok = True)


    where = (m200c_sp < np.power(10,mf+0.5)) & (m200c_sp >= np.power(10,mf)) & (gasmass_center[:,0] > 50) & (gasmass_center[:,0] < 950) & (gasmass_center[:,1] > 50) & (gasmass_center[:,1] < 950) & (gasmass_center[:,2] > 50) & (gasmass_center[:,2] < 950)
    print(np.sum(where))
    if np.sum(where)>halonum:
        halo_rands = np.random.randint(np.sum(where), size=halonum)
        halo_sel_ids = soap_ids[where][halo_rands]
    else:
        halo_sel_ids = soap_ids[where]
    
    ## not repeat calculate previous results
    #halo_sel_ids = halo_sel_ids[29:]
    print(len(halo_sel_ids))
    output = {}
    output['halo_ids'] = halo_sel_ids
    output['r200c'] = r200c_sp[np.array(halo_sel_ids-1, dtype = int)]
    output['x_gasmass_center'] = gasmass_center[:,0][np.array(halo_sel_ids-1, dtype = int)]
    output['y_gasmass_center'] = gasmass_center[:,1][np.array(halo_sel_ids-1, dtype = int)]
    output['z_gasmass_center'] = gasmass_center[:,2][np.array(halo_sel_ids-1, dtype = int)]
    output['xray_bol_emilum'] = xray_bol_emilum[np.array(halo_sel_ids-1, dtype = int)][:,0]
    output['xray_bol_phlum'] = xray_bol_phlum[np.array(halo_sel_ids-1, dtype = int)][:,0]
    output['fe17'] = np.zeros(len(halo_sel_ids))
    output['o7f'] = np.zeros(len(halo_sel_ids))
    output['o8'] = np.zeros(len(halo_sel_ids))

    
    # ######## for test ##########
    # halodoc = {}
    # index = int(halo_sel_ids[29]-1)
    # halodoc['o7f'], halodoc['o8'], halodoc['fe17'], halodoc['jointmsk'], halodoc['part_masses'], halodoc['part_dens'], halodoc['nH_dens'], halodoc['part_temperatures'], halodoc['abun_hydrogen'], halodoc['abun_helium'], halodoc['abun_carbon'], halodoc['abun_nitrogen'], halodoc['abun_oxygen'], halodoc['abun_neon'], halodoc['abun_magnesium'], halodoc['abun_silicon'], halodoc['abun_iron'],halodoc['part_xcoords'], halodoc['part_ycoords'], halodoc['part_zcoords'], halodoc['tot_abun_to_solar']  = cal_halo_summass(index)
    # output['o7f'][0], output['o8'][0], output['fe17'][0] = np.nansum(halodoc['o7f'][halodoc['jointmsk']]), np.nansum(halodoc['o8'][halodoc['jointmsk']]), np.nansum(halodoc['fe17'][halodoc['jointmsk']])
    # df1 = pd.DataFrame.from_dict(halodoc)
    # df1.to_csv(f'{savepath}/xray_linelum_snapshot{snapnum}_halo{np.array(halo_sel_ids-1, dtype = int)[0]}_partlum.csv')  
    ######### formal ###########
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        for i, result in enumerate(executor.map(cal_halo_summass, np.array(halo_sel_ids-1, dtype = int))):
            halodoc = {}
            halodoc['o7f'], halodoc['o8'], halodoc['fe17'], halodoc['jointmsk'], halodoc['part_masses'], halodoc['part_dens'], halodoc['nH_dens'], halodoc['part_temperatures'], halodoc['abun_hydrogen'], halodoc['abun_helium'], halodoc['abun_carbon'], halodoc['abun_nitrogen'], halodoc['abun_oxygen'], halodoc['abun_neon'], halodoc['abun_magnesium'], halodoc['abun_silicon'], halodoc['abun_iron'],halodoc['part_xcoords'], halodoc['part_ycoords'], halodoc['part_zcoords'], halodoc['tot_abun_to_solar'] = result
            output['o7f'][i], output['o8'][i], output['fe17'][i] = np.nansum(halodoc['o7f'][halodoc['jointmsk']]), np.nansum(halodoc['o8'][halodoc['jointmsk']]), np.nansum(halodoc['fe17'][halodoc['jointmsk']])
            df1 = pd.DataFrame.from_dict(halodoc)
            df1.to_csv(f'{savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halo{np.array(halo_sel_ids-1, dtype = int)[i]}_partlum.csv')  
            print(f'{datetime.now()} {halo_sel_ids[i]}.csv')


finish = time.perf_counter()
print(f'Finished in {(finish-start)/60:.2f} min(s)')