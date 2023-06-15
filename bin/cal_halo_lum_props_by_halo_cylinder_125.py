'''
1. Select the halo center based on soap cat, select 5 halos for each mass bin
2. Based on halo center select corresponding smallest box in snapshots
3. select only particles in r200c (numba), calculate their lum and save
(4. sum up lum of each halo, make the plot)
abundance to solar: wherea calcium and sulfur are the same as the silicon
['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'calcium', 'sulfur', 'iron']
'''
### for ids: 1. for The single halo: id in filename +1 = soap['VR/id'] 2. for the ids in sumfile: id = soap['VR/id']
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
from interpolate_X_Ray_seperate_lines_cont import interpolate_X_Ray as interp_xray
from interpolate_X_Ray_seperate_lines_cont import interpolate
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
     
def compute_lum(interp_rest_range, data, table_type, z, where, nH_dens):
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
    # in one snapshot, all the particles have the same redshift! 
    # However particles' luminosity distances are different since it also due to their z axis los distance
    data.gas.redshifts = np.ones(len(data.gas.densities))*z
    lum, __, abun_to_solar = interp_xray(
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
        return np.zeros(len(data.gas.densities)), np.zeros(len(data.gas.densities)), np.zeros(len(data.gas.densities))
    else:
        lum = lum.transpose()
        a_agn = data.gas.last_agnfeedback_scale_factors
        a_agn[a_agn==-1] = np.nan
        if np.sum(np.isfinite(a_agn))!=0:
            a_agn = np.array(a_agn)
            t_par = DESyr3.lookback_time(z)
            t_agn = DESyr3.lookback_time(1/a_agn-1)

        jointmsk = ~(((nH_dens.value<1e-8) | (nH_dens.value>1e6)) | ((data.gas.temperatures.value<1e5) | (data.gas.temperatures.value>np.power(10,9.5))) | ((abs(t_agn-t_par).value*1000<15) & (~np.isnan(a_agn))& ((np.power(10, 6.94455)<data.gas.temperatures.value) | (np.power(10, 8.24455)>data.gas.temperatures.value))))
        print(np.sum(jointmsk))
        return lum[0], jointmsk, abun_to_solar

def cal_halo_summass(sid):
    # load region
    mask = sw.mask(filename)
    boxsize = mask.metadata.boxsize
    load_region = [[(gasmass_center[:,0][sid]/boxsize[0].value-0.025)*boxsize[0], (gasmass_center[:,0][sid]/boxsize[0].value+0.025)*boxsize[0]],[(gasmass_center[:,1][sid]/boxsize[1].value-0.025)*boxsize[1], (gasmass_center[:,1][sid]/boxsize[1].value+0.025)*boxsize[1]],[(gasmass_center[:,2][sid]/boxsize[2].value-0.025)*boxsize[2], (gasmass_center[:,2][sid]/boxsize[2].value+0.025)*boxsize[2]]]
    mask.constrain_spatial(load_region)
    data = sw.load(filename, mask=mask)
    msk = msk_in_cylinder(np.array(data.gas.coordinates), (gasmass_center[:,0][sid],gasmass_center[:,1][sid],gasmass_center[:,2][sid]), 3.5, 6.25)
    nH_densities = interpdens2nH(data.gas.densities, data.gas.smoothed_element_mass_fractions.hydrogen, np.zeros(data.gas.densities.shape))
    linesbins = {'fe17':[0.724, 0.726],'o7f':[0.574,0.576],'o8':[0.653,0.656]}
    lumdict = {}
    for line in linesbins.keys():
        lumdict[line], jointmsk, abun_to_solar = compute_lum(linesbins[line], data, 'lines', reds, msk, nH_densities)
    return lumdict['o7f'], lumdict['o8'], lumdict['fe17'], jointmsk, data.gas.masses, data.gas.densities, nH_densities, data.gas.temperatures, abun_to_solar[:,0], abun_to_solar[:,1], abun_to_solar[:,2], abun_to_solar[:,3], abun_to_solar[:,4], abun_to_solar[:,5], abun_to_solar[:,6], abun_to_solar[:,7], abun_to_solar[:,10], data.gas.coordinates[:,0], data.gas.coordinates[:,1], data.gas.coordinates[:,2]

def cal_halo_summass_woxray(sid):
    # load region
    mask = sw.mask(filename)
    boxsize = mask.metadata.boxsize
    load_region = [[(gasmass_center[:,0][sid]/boxsize[0].value-0.025)*boxsize[0], (gasmass_center[:,0][sid]/boxsize[0].value+0.025)*boxsize[0]],[(gasmass_center[:,1][sid]/boxsize[1].value-0.025)*boxsize[1], (gasmass_center[:,1][sid]/boxsize[1].value+0.025)*boxsize[1]],[(gasmass_center[:,2][sid]/boxsize[2].value-0.025)*boxsize[2], (gasmass_center[:,2][sid]/boxsize[2].value+0.025)*boxsize[2]]]
    mask.constrain_spatial(load_region)
    data = sw.load(filename, mask=mask)
    msk = msk_in_cylinder(np.array(data.gas.coordinates), (gasmass_center[:,0][sid],gasmass_center[:,1][sid],gasmass_center[:,2][sid]), 3.5, 6.25)
    nH_densities = interpdens2nH(data.gas.densities, data.gas.smoothed_element_mass_fractions.hydrogen, np.zeros(data.gas.densities.shape))

    return data.gas.masses, data.gas.densities, nH_densities, data.gas.temperatures

start = time.perf_counter()
# load soap cat
print('loading soap cat...')
# Caution!!! only redshift 0 soap halo position is correct, for larger redshifts halo postions are wrong!

reds = 0.1
with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/L1000N3600/HYDRO_FIDUCIAL/SOAP/halo_properties_00{int(78-reds/0.05)}.hdf5", 'r') as catalogue_soap:
    soap_ids = np.array(catalogue_soap["VR/ID"][()])
    m200c_sp = catalogue_soap["SO/200_crit/TotalMass"][()]
    r200c_sp = catalogue_soap["SO/200_crit/SORadius"][()]
    gasmass_center = catalogue_soap["SO/200_crit/GasCentreOfMass"][()]
    # mass_center = catalogue_soap["SO/200_crit/CentreOfMass"][()] # mass center and gasmass center are similar
    # BH_center = catalogue_soap["SO/200_crit/MostMassiveBlackHolePosition"][()]  # not right
    xray_bol_emilum = catalogue_soap["SO/200_crit/XRayLuminosityWithoutRecentAGNHeating"][()]
    xray_bol_phlum = catalogue_soap["SO/200_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][()]


# define snapshot file

workpath = '/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot'
# filename = f'{workpath}/test/test_dataset/flamingo_snapshot_noxraypart.hdf5'
filename = f'/cosma8/data/dp004/flamingo/Runs/L1000N3600/HYDRO_FIDUCIAL/snapshots/flamingo_00{int(78-reds/0.05)}/flamingo_00{int(78-reds/0.05)}.hdf5'


# preload table
interp = interpolate()
print('loading table')
interp.load_table()
print('table_loaded')

np.random.seed(0)
mass_filter = np.array([12.5])
halonum = 128
for mf in mass_filter:
    where = (m200c_sp < np.power(10,mf+0.5)) & (m200c_sp >= np.power(10,mf)) & (gasmass_center[:,0] > 50) & (gasmass_center[:,0] < 950) & (gasmass_center[:,1] > 50) & (gasmass_center[:,1] < 950) & (gasmass_center[:,2] > 50) & (gasmass_center[:,2] < 950)
    print(np.sum(where))
    if np.sum(where)>halonum:
        halo_rands = np.random.randint(np.sum(where), size=halonum)
        halo_sel_ids = soap_ids[where][halo_rands]
    else:
        halo_sel_ids = soap_ids[where]
    
    
    ## not repeat calculate previous results
    # halo_sel_ids = halo_sel_ids[29:]
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

    savepath = f'{workpath}/results/xraysb_csvs_230615_{mf}_groups_1028halos_cyl'
    os.makedirs(savepath, exist_ok = True)
    # ######## for test ##########
    # halodoc = {}
    # index = int(halo_sel_ids[30]-1)
    # halodoc['o7f'], halodoc['o8'], halodoc['fe17'], halodoc['jointmsk'], halodoc['part_masses'], halodoc['part_dens'], halodoc['nH_dens'], halodoc['part_temperatures'], halodoc['abun_hydrogen'], halodoc['abun_helium'], halodoc['abun_carbon'], halodoc['abun_nitrogen'], halodoc['abun_oxygen'], halodoc['abun_neon'], halodoc['abun_magnesium'], halodoc['abun_silicon'], halodoc['abun_iron'],halodoc['part_xcoords'], halodoc['part_ycoords'], halodoc['part_zcoords'] = cal_halo_summass(index)
    # output['o7f'][0], output['o8'][0], output['fe17'][0] = np.nansum(halodoc['o7f'][halodoc['jointmsk']]), np.nansum(halodoc['o8'][halodoc['jointmsk']]), np.nansum(halodoc['fe17'][halodoc['jointmsk']])
    # df1 = pd.DataFrame.from_dict(halodoc)
    # df1.to_csv(f'{savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halo{np.array(halo_sel_ids-1, dtype = int)[0]}_partlum.csv') 
    # print('csv has been saved!')
    # df = pd.DataFrame.from_dict(output)
    # df.to_csv(f'{savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv')
    # print(f'{datetime.now()}: {savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv has been saved! ')
    # ######## for test2 ##########
    # halodoc = {}
    # index = int(halo_sel_ids[30]-1)
    # halodoc['part_masses'], halodoc['part_dens'], halodoc['nH_dens'], halodoc['part_temperatures'] = cal_halo_summass_woxray(index)
    # df1 = pd.DataFrame.from_dict(halodoc)
    # df1.to_csv(f'{savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halo{np.array(halo_sel_ids-1, dtype = int)[0]}_partlum.csv') 
    # print('halo csv has been saved!')
    # df = pd.DataFrame.from_dict(output)
    # df.to_csv(f'{savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv')
    # print(f'{savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv has been saved! ')
    # ###### further test ##########
    # with concurrent.futures.ProcessPoolExecutor(1) as executor:
    #     for i, result in enumerate(executor.map(cal_halo_summass_woxray, np.array(halo_sel_ids-1, dtype = int))):
    #         print(i)
    #         halodoc = {}
    #         halodoc['part_masses'], halodoc['part_dens'], halodoc['nH_dens'], halodoc['part_temperatures'] = result
    #         print(f'{datetime.now()} {halo_sel_ids[i]}.csv')
    #         df1 = pd.DataFrame.from_dict(halodoc)
    #         df1.to_csv(f'{savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halo{np.array(halo_sel_ids-1, dtype = int)[i]}_partlum.csv')  
            
    # df = pd.DataFrame.from_dict(output)
    # df.to_csv(f'{savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv')
    # print(f'{datetime.now()}: {savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv has been saved! ')

    ######### formal ###########
    with concurrent.futures.ProcessPoolExecutor(16) as executor:
        for i, result in enumerate(executor.map(cal_halo_summass, np.array(halo_sel_ids-1, dtype = int))):
            halodoc = {}
            halodoc['o7f'], halodoc['o8'], halodoc['fe17'], halodoc['jointmsk'], halodoc['part_masses'], halodoc['part_dens'], halodoc['nH_dens'], halodoc['part_temperatures'], halodoc['abun_hydrogen'], halodoc['abun_helium'], halodoc['abun_carbon'], halodoc['abun_nitrogen'], halodoc['abun_oxygen'], halodoc['abun_neon'], halodoc['abun_magnesium'], halodoc['abun_silicon'], halodoc['abun_iron'],halodoc['part_xcoords'], halodoc['part_ycoords'], halodoc['part_zcoords'] = result
            output['o7f'][i], output['o8'][i], output['fe17'][i] = np.nansum(halodoc['o7f'][halodoc['jointmsk']]), np.nansum(halodoc['o8'][halodoc['jointmsk']]), np.nansum(halodoc['fe17'][halodoc['jointmsk']])
            df1 = pd.DataFrame.from_dict(halodoc)
            df1.to_csv(f'{savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halo{np.array(halo_sel_ids-1, dtype = int)[i]}_partlum.csv')  
            print(f'{datetime.now()} {halo_sel_ids[i]}.csv')
    df = pd.DataFrame.from_dict(output)
    df.to_csv(f'{savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv')
    print(f'{datetime.now()}: {savepath}/xray_linelum_snapshot{int(78-reds/0.05)}_halomass_btw_{int(mf*10)}_{int((mf+0.5)*10)}.csv has been saved! ')

finish = time.perf_counter()
print(f'Finished in {(finish-start)/60:.2f} min(s)')