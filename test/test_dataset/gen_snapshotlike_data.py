'''
Generate snapshot like data
with only several particle bright in X-ray


ERROR:
Unfortunately, this doesn't work , it will report error when input in interpolate code because it doesn't have units?
'''
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import swiftsimio as sw


# create a snapshot hdf5 file where there is no x-ray lum particles
# therefore set the temperature and densities to 0

import h5py

filepath = '/cosma7/data/dp004/dc-chen3/roi_files/flamingo_0008.hdf5'
new_file_path = 'flamingo_snapshot_noxraypart.hdf5'

# Open the existing HDF5 file in read mode
with h5py.File(filepath, 'r') as old_file:
    # Create a new HDF5 file in write mode
    with h5py.File(new_file_path, 'w') as new_file:
        # Copy datasets from the existing file to the new file
        for dataset_name, dataset in old_file['GasParticles'].items():
            print(dataset_name, dataset)
            if dataset_name == 'Temperatures':
                new_file.create_dataset(f'GasParticles/{dataset_name}', data = np.zeros(50))
            else:
                new_file.create_dataset(f'GasParticles/{dataset_name}', data=dataset[()][:50])
            for attr_name, attr_data in old_file[f'GasParticles/{dataset_name}'].attrs.items():
                new_file[f'GasParticles/{dataset_name}'].attrs[attr_name] = attr_data


# # # The new file is saved with the updated dataset values

# # # load smallest part of swift data, using roi file
# # old_file_path = f'/cosma7/data/dp004/dc-chen3/roi_files/flamingo_0008.hdf5'
# # new_file_path = 'flamingo_snapshot_noxraypart.hdf5'
# # # os.system(f'rm {new_file_path}')
# # os.system(f'cp {old_file_path} {new_file_path}')
# # with h5py.File(new_file_path, 'r+') as f:
# #     f['GasParticles/Temperatures'][()] = np.zeros(f['GasParticles/Temperatures'][()].shape)
# #     f.flush()






