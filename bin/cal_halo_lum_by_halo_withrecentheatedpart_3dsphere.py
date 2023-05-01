'''
1. Select the halo center based on soap cat, select 5 halos for each mass bin
2. Based on halo center select corresponding smallest box in snapshots
3. select only particles in r200c (numba), calculate their lum and save
(4. sum up lum of each halo, make the plot)
'''
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import swiftsimio as sw
import pandas as pd
import h5py
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
from interpolate_X_Ray_seperate_lines_cont import interpolate_X_Ray as interp_xray
from interpolate_X_Ray_seperate_lines_cont import interpolate
from unyt import g, cm, mp
import concurrent.futures
import time


m_nu = [0.02, 0.02, 0.02] * u.eV
DESyr3 = FlatLambdaCDM(H0=68.1, Om0=0.3046, m_nu=m_nu, Ob0=0.0486, Tcmb0=2.725)


@nb.jit(nopython=True)
def halo_part_in_r200c_nb(coor, halo_center, r200c):
    n = coor.shape[1]
    where = np.empty(coor.shape[0], dtype=np.bool_)
    for i in range(coor.shape[0]):
        d2 = 0.0
        for j in range(n):
            d2 += (coor[i,j] - halo_center[j])**2
        where[i] = d2 < r200c**2
    return where
def interpdens2nH(interpdens, hydrogen_massfrac, redshifts):
    # after np.log10, unyt unit is lost
    scale_factor = 1/(redshifts+1)
    return interpdens.to(g/cm**3) * hydrogen_massfrac / mp.to(g) * (1/scale_factor**3) 
def compute_lum(interp_rest_range, data, table_type, z, where):
    """
    This function compute xray luminosity for particle fall in observed energy bin at z=0

    Parameters
    -----------------------
    interp_rest_range: np.array 1x2
        the energy range to interpolate a spectrum

    data: snapshot data structure 
        snapshot data structure updated by particle lightcone's data
    
    table_type: table_type to interpolate
        ["all", "lines", "cont"]

    Returns
    -----------------------
    lum_arr: np.array shape:(energy_bins, particle_nums)
        luminosities for each particle 
    
    """
    print("interpolating xrays")
    data.gas.redshifts = np.ones(len(data.gas.densities))*z
    lum, restframe_energy = interp_xray(
        data.gas.densities,
        data.gas.temperatures,
        data.gas.smoothed_element_mass_fractions,
        data.gas.redshifts,
        data.gas.masses,
        interp,
        where,
        fill_value = 0,
        bin_energy_lims=interp_rest_range,
        table_type=table_type,
    )
    if lum is False:
        print('halo not bright in xray')
        return np.zeros(3)
    else:
        lum = lum.transpose()
        nH_dens = interpdens2nH(data.gas.densities, data.gas.smoothed_element_mass_fractions.hydrogen, data.gas.redshifts)
        lum[:,(nH_dens.value<1e-8) | (nH_dens.value>1e6)] = 0
        lum[:,(data.gas.temperatures.value<1e5) | (data.gas.temperatures.value>np.power(10,9.5))] = 0
        return lum[0]

start = time.perf_counter()
# load soap cat
print('loading soap cat...')
# Caution!!! only redshift 0 soap halo position is correct, for larger redshifts halo postions are wrong!
reds = 0
with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/SOAP/halo_properties_00{int(77-reds/0.05)}.hdf5", 'r') as catalogue_soap:
    soap_ids = np.array(catalogue_soap["VR/ID"][()])
    m200c_sp = catalogue_soap["SO/200_crit/TotalMass"][()]
    r200c_sp = catalogue_soap["SO/200_crit/SORadius"][()]
    gasmass_center = catalogue_soap["SO/200_crit/GasCentreOfMass"][()]
    # mass_center = catalogue_soap["SO/200_crit/CentreOfMass"][()] # mass center and gasmass center are similar
    # BH_center = catalogue_soap["SO/200_crit/MostMassiveBlackHolePosition"][()]  # not right
    xray_bol_emilum = catalogue_soap["SO/200_crit/XRayLuminosityWithoutRecentAGNHeating"][()]
    xray_bol_phlum = catalogue_soap["SO/200_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][()]


# define snapshot file
filename = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_00{int(77-reds/0.05)}/flamingo_00{int(77-reds/0.05)}.hdf5'
workpath = '/cosma8/data/dp004/dc-chen3/work/bin/gen_xray_pipeline/230318/cal_halo_lum_by_halo_230331'


# preload table
interp = interpolate()
print('loading table')
interp.load_table()
print('table_loaded')

np.random.seed(19680801)
mass_filter = np.array([13, 13.5, 14, 14.5, 15])
for mf in mass_filter[::-1]:
    print(mf)
    where = (m200c_sp < np.power(10,mf+0.5)) & (m200c_sp >= np.power(10,mf)) & (gasmass_center[:,0] > 50) & (gasmass_center[:,0] < 950) & (gasmass_center[:,1] > 50) & (gasmass_center[:,1] < 950) & (gasmass_center[:,2] > 50) & (gasmass_center[:,2] < 950)
    if np.sum(where)>4:
        halo_rands = np.random.randint(np.sum(where), size=4)
        halo_sel_ids = soap_ids[where][halo_rands]
    else:
        halo_sel_ids = soap_ids[where]
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
    

    def cal_halo_summass(sid):
        xc, yc, zc = gasmass_center[sid]
        # load region
        mask = sw.mask(filename)
        boxsize = mask.metadata.boxsize
        load_region = [[(gasmass_center[:,0][sid]/boxsize[0].value-0.025)*boxsize[0], (gasmass_center[:,0][sid]/boxsize[0].value+0.025)*boxsize[0]],[(gasmass_center[:,1][sid]/boxsize[1].value-0.025)*boxsize[1], (gasmass_center[:,1][sid]/boxsize[1].value+0.025)*boxsize[1]],[(gasmass_center[:,2][sid]/boxsize[2].value-0.025)*boxsize[2], (gasmass_center[:,2][sid]/boxsize[2].value+0.025)*boxsize[2]]]
        mask.constrain_spatial(load_region)
        data = sw.load(filename, mask=mask)
        msk = halo_part_in_r200c_nb(np.array(data.gas.coordinates), (gasmass_center[:,0][sid],gasmass_center[:,1][sid],gasmass_center[:,2][sid]), r200c_sp[sid])
        linesbins = {'fe17':[0.715,0.717],'o7f':[0.574,0.576],'o8':[0.653,0.656]}
        lumdict = {}
        for line in ['fe17', 'o7f', 'o8']:
            lumdict[line] = compute_lum(linesbins[line], data, 'lines', reds, msk)
        return lumdict['o7f'], lumdict['o8'], lumdict['fe17']
    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        for i, result in enumerate(executor.map(cal_halo_summass,np.array(halo_sel_ids-1, dtype = int))):
            halodoc = {}
            halodoc['o7f'], halodoc['o8'], halodoc['fe17'] = result
            output['o7f'][i], output['o8'][i], output['fe17'][i] = np.nansum(halodoc['o7f']), np.nansum(halodoc['o8']), np.nansum(halodoc['fe17'])
            df1 = pd.DataFrame.from_dict(halodoc)
            df1.to_csv(f'{workpath}/xraylum_csvs_230404_debug_abunarray_withrecentheatedpart/xray_linelum_snapshot75_halo{np.array(halo_sel_ids-1, dtype = int)[i]}_partlum_230404.csv')
    df = pd.DataFrame.from_dict(output)
    df.to_csv(f'{workpath}/xraylum_csvs_230404_debug_abunarray_withrecentheatedpart/xray_linelum_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230404.csv')
    print(f'{workpath}/xraylum_csvs_230404_debug_abunarray_withrecentheatedpart/xray_linelum_snapshot75_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}_230404.csv has been saved! ')

finish = time.perf_counter()
print(f'Finished in {(finish-start)/60:.2f} min(s)')