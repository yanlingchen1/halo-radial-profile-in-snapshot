'''
1. Read coords from soap cat 
2. generate and check visually for all the gas particles in r200c > m200c

'''

import healpy as hp
import h5py
import matplotlib.pyplot as plt
import os
import sys
from unyt import Mpc, Msun
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import numba as nb
from tqdm import tqdm

@nb.jit(nopython=True)
def halo_part_in_r200c_nb(coor, halo_center, r200c):
    n = coor.shape[1]
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = d2 < (r200c)**2
    return where

workpath = '/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot'

reds = 0.1
# soap
nside = 1024
snapnum = int(77-reds/0.05)
m200c_filter = 15.0

print('loading soap cat...')
# first make filters via mass, then read coords and radius only based on mass
with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_00{snapnum}.hdf5", 'r') as catalogue_soap:
    mainhaloids = catalogue_soap["VR/HostHaloID"][()]
    M200_tot = catalogue_soap["SO/200_crit/TotalMass"][()]
    M200_gas = catalogue_soap["SO/200_crit/GasMass"][()]
    R200 = catalogue_soap["SO/200_crit/SORadius"][()] * (1+reds)
    # snapcoor = catalogue_soap["VR/CentreOfPotential"][()]
    snapcoor = catalogue_soap["SO/200_crit/GasCentreOfMass"][()] * (1+reds)

where = (M200_tot>np.power(10.,m200c_filter)) & (mainhaloids==-1)
M200_tot = M200_tot[where]
M200_gas = M200_gas[where]
R200 = R200[where]
snapcoor = snapcoor[where]


snapshot_datapath = '/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots_downsampled/'
snapshot_filename = f'{snapshot_datapath}/flamingo_00{int(77-reds/0.05)}.hdf5'
workpath = '/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot'
savepath = f'{workpath}/test/png'

print('loading snapshot data')
# load snapshot data
from swiftsimio import load
data = load(snapshot_filename)
print(data.gas.masses.units)
halo_masses = []
for id in tqdm(range(len(R200))):
    where = halo_part_in_r200c_nb(data.gas.coordinates.value, snapcoor[id], R200[id])
    halo_masses.append(np.sum(data.gas.masses[where]))

plt.scatter(np.array(halo_masses)*1e10, M200_gas, alpha = 0.5)
plt.xlabel('total gas mass in r200 in snapshot [$\\rm M_{\odot}$]')
plt.ylabel('M200_gas in SOAP')
plt.xscale('log')
plt.title(f'z={reds}, halo mass > $\\rm 10^{{15}} M_{{\odot}}$')
plt.yscale('log')
plt.plot(np.array(halo_masses)*1e10,np.array(halo_masses)*1e10, c = 'k')
plt.legend()
plt.savefig(f'{workpath}/fig/test/halo_pos_test_L1000N1800_z01_gasmasscenter.png')


# from velociraptor import load as vl_load
# catalog_redshift = 0
# cat_ind = str(int(77-np.round(catalog_redshift/0.05)))
# cata_loc="/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/VR/"
# cata_name=cata_loc+'catalogue_00'+cat_ind+'/vr_catalogue_00'+cat_ind+'.properties.0'
# catalogue = vl_load(cata_name)
# halo_id = np.arange(len(catalogue.ids.hosthaloid.value))[(catalogue.ids.hosthaloid.value==-1) & (catalogue.masses.mass_200crit.value>np.power(10.,4.5))]# & (catalogue.masses.mass_200crit.value<1e5)]
# xcmbp= catalogue.positions.xcmbp.to(Mpc).value[halo_id]
# ycmbp= catalogue.positions.ycmbp.to(Mpc).value[halo_id]
# zcmbp= catalogue.positions.zcmbp.to(Mpc).value[halo_id]
# r200c = catalogue.radii.r_200crit.to(Mpc).value[halo_id]
# r200c_lst = []
# for k in range(len(xcmbp)):
#     r200c_pix = r200c/(1000/1024)
#     i,j = cartproj.xy2ij(np.array([xcmbp[k], ycmbp[k]]))
#     c = plt.Circle((j,i), r200c_pix, fill=False, color='white')
#     r200c_lst.append(r200c_pix)
# r200c_lst = np.array(r200c_lst)
# print('vr:', r200c_lst.min(), r200c_lst.max(), np.median(r200c_lst))



