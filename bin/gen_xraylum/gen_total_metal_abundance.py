'''
This program generate total metal abundance for every particle
1. load snapshot file, [SmoothedMetalMassFractions]
2. obey all the procedures in cal_halo_lum*L1000N1800.py
3. Convert the metal mass fraction to Metal abundance
4. save separate file with only the metal

'''

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


# count time
start = time.perf_counter()

# Caution!!! only redshift 0 soap halo position is correct, for larger redshifts halo postions are wrong!

# define the redshift and the halo mass
reds = 0.1

sim = 'L1000N1800'
snapnum = int(77-reds/0.05)
halonum = 1028

# sim = 'L1000N3600'
# snapnum = int(78-reds/0.05)
# halonum = 128

np.random.seed(0)
mass_filter = np.array([13.5])


# load soap cat
print('loading soap cat...')
with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/{sim}/HYDRO_FIDUCIAL/SOAP/halo_properties_00{snapnum}.hdf5", 'r') as catalogue_soap:
    soap_ids = np.array(catalogue_soap["VR/ID"][()])
    m200c_sp = catalogue_soap["SO/200_crit/TotalMass"][()]
    r200c_sp = catalogue_soap["SO/200_crit/SORadius"][()] * (1+reds)
    gasmass_center = catalogue_soap["SO/200_crit/GasCentreOfMass"][()] * (1+reds)
    # mass_center = catalogue_soap["SO/200_crit/CentreOfMass"][()] # mass center and gasmass center are similar
    # BH_center = catalogue_soap["SO/200_crit/MostMassiveBlackHolePosition"][()]  # not right
    xray_bol_emilum = catalogue_soap["SO/200_crit/XRayLuminosityWithoutRecentAGNHeating"][()]
    xray_bol_phlum = catalogue_soap["SO/200_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][()]

def cal_halo_summass(sid):
    # load snapshot file
    filename = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_00{snapnum}/flamingo_00{snapnum}.hdf5'
    mask = sw.mask(filename)
    boxsize = mask.metadata.boxsize
    load_region = [[(gasmass_center[:,0][sid]/boxsize[0].value-0.025)*boxsize[0], (gasmass_center[:,0][sid]/boxsize[0].value+0.025)*boxsize[0]],[(gasmass_center[:,1][sid]/boxsize[1].value-0.025)*boxsize[1], (gasmass_center[:,1][sid]/boxsize[1].value+0.025)*boxsize[1]],[(gasmass_center[:,2][sid]/boxsize[2].value-0.025)*boxsize[2], (gasmass_center[:,2][sid]/boxsize[2].value+0.025)*boxsize[2]]]
    mask.constrain_spatial(load_region)
    data = sw.load(filename, mask=mask)
    msk = msk_in_cylinder(np.array(data.gas.coordinates), (gasmass_center[:,0][sid],gasmass_center[:,1][sid],gasmass_center[:,2][sid]), 5*r200c_sp, 6.25)

    # According to FLAMINGO paper, total metal mass fraction is in unit of 0.0134 solar abundance, therefore the abundance to solar is 1/0.0134?
    tot_abun = data.gas.smoothed_metal_mass_fractions[msk]/0.0134

    return data.gas.particle_ids[msk], tot_abun


# define paths
mf = 13.5
workpath = '/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot'
savepath = f'{workpath}/results/redshift_01/{sim}/xraysb_csvs_230718_{mf}_groups_1028halos_cyl'
os.makedirs(savepath, exist_ok = True)

### select halo id
where = (m200c_sp < np.power(10,mf+0.5)) & (m200c_sp >= np.power(10,mf)) & (gasmass_center[:,0] > 50) & (gasmass_center[:,0] < 950) & (gasmass_center[:,1] > 50) & (gasmass_center[:,1] < 950) & (gasmass_center[:,2] > 50) & (gasmass_center[:,2] < 950)
print(np.sum(where))
if np.sum(where)>halonum:
    halo_rands = np.random.randint(np.sum(where), size=halonum)
    halo_sel_ids = soap_ids[where][halo_rands]
else:
    halo_sel_ids = soap_ids[where]

# initialize
output = {}
for header in ['part_xcoords', 'part_ycoords', 'part_zcoords', 'tot_abun']:
    output[header] = np.zeros(len(halo_sel_ids))
    
for sid in (halo_sel_ids-1):
    output['part_xcoords'], output['part_ycoords'], output['part_zcoords'],  output['tot_abun'] = cal_halo_summass(sid)

    df = pd.DataFrame.from_dict(output)
    df.to_csv(f'{savepath}/tot_abun_snapshot{snapnum}_halo{sid}.csv')

    print(f'{datetime.now()}: csv has been saved!')

print(f'{datetime.now()}: Program successfully finished!')
