from swiftsimio import load
from swiftsimio.visualisation.projection import project_gas
import datetime
import os
from unyt import msun, kpc
import numpy as np

current_time = datetime.datetime.now().strftime("%H:%M:%S")
print("Begins:", current_time)

reds = 0.1
snapshot_datapath = '/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots_downsampled/'
snapshot_filename = f'{snapshot_datapath}/flamingo_00{int(77-reds/0.05)}.hdf5'
workpath = '/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot'
savepath = f'{workpath}/test/png'
print('loading data...')
data = load(snapshot_filename)

# This creates a grid that has units msun / Mpc^2, and can be transformed like
# any other unyt quantity
print('making plot...')
size = 1000 * kpc
test_size = 10 * kpc
region = [test_size, test_size, test_size, test_size] 
mass_map = project_gas(data, resolution=1024, project="masses", region = region, parallel=True)

# Let's say we wish to save it as msun / kpc^2,
mass_map.convert_to_units(msun / kpc**2)

from matplotlib.pyplot import imsave
from matplotlib.colors import LogNorm
print('plotting...')
# Normalize and save
imsave(f"gas_surface_dens_map_snapshot_reds{reds}_size{test_size}kpc.png", LogNorm()(mass_map.value), cmap="viridis")

current_time = datetime.datetime.now().strftime("%H:%M:%S")
print("Finishes:", current_time)