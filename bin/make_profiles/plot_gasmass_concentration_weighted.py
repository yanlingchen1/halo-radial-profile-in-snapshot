import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py
from glob import glob
from datetime import datetime


### The densities (nH or particles) are linear, not in log!
# define functions
def filter_prof_cts(prof, thres, ctsprof):
    if len(ctsprof) == len(prof)+1:
        ctsprof = ctsprof[:-1]
        prof[ctsprof<thres] = np.nan
    elif len(ctsprof) == len(prof):
        prof[ctsprof<thres] = np.nan
    else:
        raise ValueError(f'Wrong profile length! ctsprof: {ctsprof.shape}, prof: {prof.shape}')
    return prof

def argmedian(data):
    return np.argsort(data)[:,len(data)//2]

def basic_figure_style():
    SMALL_SIZE = 6*2                                    
    MEDIUM_SIZE = 8*2
    BIGGER_SIZE = 10*2

    plt.rc('font', size=MEDIUM_SIZE, family='serif')          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)                     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)                    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE, direction='out')    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE, direction='out')    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)                    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)                  # fontsize of the figure title
    plt.rc('lines', linewidth=2) 
    plt.rc('axes', grid=True) #
    plt.rc('grid', alpha=0.7) #
    plt.rc('xtick', top=True)
    plt.rc('ytick', right=True)
    plt.rc('axes.formatter', use_mathtext=True, min_exponent=4, useoffset=False)
    # plt.rc('figure', figsize='8, 6')                         # size of the figure, used to be '4, 3' in inches
basic_figure_style()
cb = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']
    ######################################################


def choose_nastasha_profs(prop):
    '''
    prop is profname
    Only for halo mass 13.5-14.0
    
    '''
    nas_dir = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/nastasha_plots_data'
    if 'temperature' in prop:
        if 'mass' in prop:
            df = pd.read_csv(f'{nas_dir}/T_r_weighted_mass_M135-140.csv', header = None)
        elif 'o7f' in prop:
            df = pd.read_csv(f'{nas_dir}/T_r_weighted_OVIIr_M135-140.csv', header = None)
        elif 'o8' in prop:
            df = pd.read_csv(f'{nas_dir}/T_r_weighted_OVIII_M135-140.csv', header = None)
        elif 
        else:
            print('No corresponding Nastasha profiles')
            return None

    elif 'nH' in prop:
        if 'mass' in prop:
            df = pd.read_csv(f'{nas_dir}/nH_r_weighted_mass_M135-140.csv', header = None)
        elif 'o7f' in prop:
            df = pd.read_csv(f'{nas_dir}/nH_r_weighted_OVIIr_M135-140.csv', header = None)
        elif 'o8' in prop:
            df = pd.read_csv(f'{nas_dir}/nH_r_weighted_OVIII_M135-140.csv', header = None)
        else:
            print('No corresponding Nastasha profiles')
            return None

    elif 'abun' in prop:
        if 'mass' in prop:
            df = pd.read_csv(f'{nas_dir}/totabun_r_weighted_mass_M135-140.csv', header = None)
        elif 'o7f' in prop:
            df = pd.read_csv(f'{nas_dir}/totabun_r_weighted_OVIIr_M135-140.csv', header = None)
        elif 'o8' in prop:
            df = pd.read_csv(f'{nas_dir}/totabun_r_weighted_OVIII_M135-140.csv', header = None)
        else:
            print('No corresponding Nastasha profiles')
            return None
    else: 
        print('No corresponding Nastasha profiles')
        return None

    return df


# define parameters
# massfilter = np.arange(13.0, 15.5, 0.5)
massfilter = [13.5]
props = ['part_temperatures', 'nH_dens', 'abun_iron', 'abun_oxygen'] # 
units = ['K', '$\\rm cm^{-3}$', '$Z_{\odot}$', '$Z_{\odot}$']
weightings = ['part_vol'] #,'part_vol', , 'o7f', 'o8', 'fe17'


# define the sim
reds = 0.1

sim  = 'L1000N1800'
snapnum = int(77 - reds / 0.05)
halonum = 1028

# sim = 'L1000N3600'
# snapnum = int(78 - reds / 0.05)
# halonum = 128

# define plotting parameters
import seaborn as sns
cb = sns.color_palette("colorblind").as_hex()
lstyles = ['solid', 'dotted']



# define the paths
workpath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/results/redshift_01/{sim}'
savepath = f'/cosma8/data/dp004/dc-chen3/work/bin/halo-radial-profile-in-snapshot/fig/profiles_230718/weighted_profs'
os.makedirs(savepath, exist_ok=True)
# Define the radius bins
xbins_mean = np.arange(-1.5, 1, 0.25)
xbins_med = np.arange(-1.5, 1, 0.1)
xbins = {'010dex':[xbins_med, 'med']} #'025dex':[xbins_mean, 'mean'], 

for mf in massfilter:
    # define paths'010dex':[xbins_med, 'med']
    ### for old datas 
    # result_path = f'{workpath}/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_{halonum}halos'
    # old_data_path = f'{workpath}/results_other_prop_but_wrong_xrayflux/xraysb_csvs_230504_{mf}_groups_{halonum}halos'
    # mulprof_path = f'{workpath}/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_{halonum}halos/xraylum_csvs_230616_{mf}_groups_radial_pkpc_cyl'
    # weightprof_path = f'{workpath}/results_add_xraylum_sb_230509/xraysb_csvs_{mf}_groups_{halonum}halos/xraylum_csvs_230612_{mf}_groups_radial_pkpc_cyl'

    ### for new datas 
    result_path = f'{workpath}/xraysb_csvs_230718_{mf}_groups_{halonum}halos_cyl'
    old_data_path = f'{workpath}/xraysb_csvs_230718_{mf}_groups_{halonum}halos_cyl'
    mulprof_path = f'{workpath}/profiles_230718_{mf}_paratest'
    weightprof_path = f'{workpath}/profiles_230718_{mf}'

    for i, prop in enumerate(props):
        for weighting in weightings:
            for shape in ['sph']:
                fig, ax = plt.subplots(figsize = (8,8))
                for m, type in enumerate(['excl']):

                    # make a profile plot: {mf}_{prop}_{weighting}_{shape}_{recentpart_type}
                    for j, binning in enumerate(xbins.keys()):
                        # define plotting style
                        lstyle = lstyles[j]

                        # read data
                        ## read haloids from sumfile
                        sumfile = glob(f'{old_data_path}/*btw*')[0]
                        halo_ids = pd.read_csv(sumfile)['halo_ids']
                        halo_ids = np.array(halo_ids).astype(int)
                        xray_emibollum = pd.read_csv(sumfile)['xray_bol_emilum']


                        ## from soap cat read m200c and r200c
                        with h5py.File(f"/cosma8/data/dp004/flamingo/Runs/{sim}/HYDRO_FIDUCIAL/SOAP/halo_properties_00{snapnum}.hdf5", 'r') as catalogue_soap:
                            soap_ids = np.array(catalogue_soap["VR/ID"][()])
                            r200c_sp = catalogue_soap["SO/200_crit/SORadius"][()] * (1+reds)


                        ## read profiles
                        prof_name = f'{prop}_mul_{weighting}_{binning}_{type}_{shape}.csv'
                        mul_prof = pd.read_csv(f'{mulprof_path}/{prof_name}') 

                        
                        if weighting == 'vol':
                            vol_profname = f'{weightprof_path}/part_vol_{binning}_{type}_{shape}.csv'
                            weight_prof = pd.read_csv(vol_profname)
                        elif weighting == 'mass':                   
                            weight_prof = pd.read_csv(f'{weightprof_path}/part_masses_{binning}_{type}_{shape}.csv')
                        else:
                            print(f'{weightprof_path}/{weighting}_{binning}_{type}_{shape}.csv')
                            weight_prof = pd.read_csv(glob(f'{weightprof_path}/{weighting}_{binning}_{type}_{shape}.csv')[0])

                        # exclude the radii bin whose cts < 50 and cut the data of zeros rows. 
                        
                        
                        cts_prof = np.array(pd.read_csv(f'{weightprof_path}/cts_{binning}_{type}_{shape}.csv'))
                        # from IPython import embed
                        # embed()


                        mul_prof = filter_prof_cts(np.array(mul_prof), 50, cts_prof)
                        weight_prof = filter_prof_cts(np.array(weight_prof), 50, cts_prof)


                        # from IPython import embed
                        # embed()


                        # weight the profile
                        if len(weight_prof) -1  == len(mul_prof):
                            weight_prof = weight_prof[:-1]
                        else:
                            raise ValueError('Wrong mul_prof or weight_prof length!')

                        multiplied_prof = np.array(mul_prof)/np.array(weight_prof)
                        
                        # from IPython import embed
                        # embed()


                        # weighted by xray emi bol lum in r200c from soap
                        final_prof = multiplied_prof[:, 1:] / np.array(xray_emibollum) * np.nanmedian(xray_emibollum)
                        print(final_prof)
                        # from IPython import embed
                        # embed()

                        # plot profiles: plot profiles of all halos and the median and mean profile. only label 1 kind of line
                        # for k in range(final_prof.shape[1]):
                        #     if k==0:
                        #         plt.plot(xbins[binning][0]/r200c[k], final_prof[:,k], c = cb[j+2], label = f'{shape}_{binning}_{type}', alpha = 0.02, linestyle = lstyle)
                        #     else:
                        #         plt.plot(xbins[binning][0]/r200c[k], final_prof[:,k], c = cb[j+1], linestyle = lstyle, alpha = 0.1)

                        plt.plot(xbins[binning][0][:-1], np.log10(np.nanmedian(final_prof, axis=1)), c = cb[j+2], label = f'{shape}_{binning}_median_{type}', linestyle = lstyles[m], linewidth = 3, alpha = 1)
                        plt.plot(xbins[binning][0][:-1], np.log10(np.nanmean(final_prof, axis=1)), c = cb[j+1], label = f'{shape}_{binning}_mean_{type}', linestyle = lstyles[m], linewidth = 3, alpha = 1)

                        # plot relative abundance

                # plot Nastasha
                df = choose_nastasha_profs(prof_name)
                if df is not None:
                    plt.plot(df[0], df[1], c = cb[0], label = 'Nastasha')

                # plotting settings
                plt.xlabel('$\\rm log_{10}(r / r_{200c})$')
                plt.ylabel(f'$\\rm log_{{10}}$  {prop} weighted by {weighting} [{units[i]}]')
                plt.xlim(-1.5, 1.)

                plt.title(f'{halonum} halos in halo mass bin $10^{{{mf}}}$ - $10^{{{mf+0.5}}}$ at z={reds} \n {shape}-{type}')
                plt.legend()

                plt.savefig(f'{savepath}/z01_{mf}_{prop}_weightby_{weighting}_{shape}_norm_by_weightL200c.png')
                print(f'{datetime.now()}:plot has been created!')
                plt.close()


