import numpy as np
from astropy import units as u

def msk_in_sph(coor, halo_center, r1, r2):
    n = 3
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = (d2 < r2**2) & (d2 >= r1**2)
    
    return where

def generate_random_coordinates(n):
    """Generate a random set of n coordinates with x, y, z components"""
    x = np.random.uniform(low=-1, high=1, size=n)
    y = np.random.uniform(low=-1, high=1, size=n)
    z = np.random.uniform(low=-1, high=1, size=n)
    coordinates = np.vstack([x, y, z]).T * u.Mpc
    return coordinates

def test_msk_in_sph():
    """Test the msk_in_cylinder function"""
    # Generate a random set of coordinates
    coordinates = generate_random_coordinates(1000)
    print(coordinates.shape)
    # Define the center of the cylinder
    center = np.array([0.5, 0.5, 0]) * u.Mpc
    # Define the height and radius of the cylinder
    height = 0.4*u.Mpc
    r2 = 0.3*u.Mpc
    r1 = 0.2*u.Mpc

    # Filter out coordinates inside the cylinder
    filtered_coordinates = coordinates[msk_in_sph(coordinates.value, center.value, r1.value, r2.value)]
    print(type(filtered_coordinates), filtered_coordinates.shape)
    # Check that the filtered coordinates are inside the cylinder
    dx = filtered_coordinates[:,0] - center[0]
    dy = filtered_coordinates[:,1] - center[1]
    dz = filtered_coordinates[:,2] - center[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    assert np.all((r < r2) & (r>=r1))  

    print("All tests passed!")

test_msk_in_sph()