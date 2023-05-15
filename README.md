# halo-radial-profile-in-snapshot

## Data
Flamingo snapshots (L1000N1800) median resolution (particle mass ~1e9 solar mass)

## Usage 
- Plot halo radial profile for various properties of different halo mass bins.
- Plot halo L200c versus halo masses, and compared with Nastasha for sanity check.

## STEPS
### STEP1: generate particle xray flux in cylinder
For every halo, calculate the xray flux for every particles in the cylinder.
- The xray flux of every particle is defined as $L_{part}/(4*\pi * d_L^2)$, where luminosity distance $d_L = coordinate_z * (1+redshift)$ 
- The cylinder center is halo's gas mass center. The cylinder length is 6.25 cMpc, the radius of the cylinder is 3.5  cMpc. Defined as the same as Nastasha[1].
- The recently heated particles are excluded. 
- Note: I took halos in snapshot z=0 while Nastasha took halos in snapshot z=0.1.
sbatch cal_halo_xraysb.sh

Using this data can generate [xray sb profile (sb vs halo radius)] or [xray sum sb vs halo masses]

### STEP2: xray sb halo profile
1. I divided the cylinder into azimuthal cylinder slices (radial bin) along the line of sight (along z axis).  In each slice I sum up the Xray luminosities of particles. 
2. In every radial bin I take the mean of all the halos in this bin to make mean profile. 
3. In every radial bin I take the median of halos whose particle counts in this bin >80. And I cut off the median profile if the bin has <90% halos whose have particle counts >80.  
sbatch cal_halo_xraysb_profile.sh (need parallel since it takes resources to filter particles in bins)
python3 plot_xray_radial_profile.py

### STEP3: xray sum lum vs halo masses
Right approach:
Take sum luminosity of particles of halo in r200c, divide by the same luminosity distance for every particle, which is 1.48e27 cm.


Wrong approach:
For every cylinder defined in step1, I sum up the xray sb of the particles in this cylinder. Then I convert the units to be the same as Nastasha's unit.
python3 plot_halomass_vs_xraylum.py (The direct sum line xray lum of output table is in cylinder.)

Reference
[1] https://ui.adsabs.harvard.edu/abs/2021arXiv210804847W

## 2023-05-09 Quick fix
- Bug: 
1. Just notice nastasha xraylum vs halo mass use 3d profile, and fixed lum distance
2. Just notice nastasha xray sb vs radii us luminosity distance based on the 6.25 cMpc slice,
I shouldn't have applied the whole box z as lum distance.

- Quick fix:
1. reverse the flux to lum, and divide by correct lum dist
2. calculate sum lum in r200c instead of in the whole cylinder.
Therefore interpolate xray can be avoided, which should be much faster.

- add update_xraylum.py file, save to results: lum and corrected flux


## 2023-05-15 
Nastasha included the recently heated particles. Perhaps she used fixed luminosity disatance 
for halo SB.

## STEPS
### STEP1: generate particle xray flux in cylinder
For every halo, calculate the xray flux for every particles in the cylinder.
- The xray flux of every particle is defined as $L_{part}/(4*\pi * d_L^2)$, where luminosity distance $d_L = 1.48e27 cm$ 
- The cylinder center is halo's gas mass center. The cylinder length is 6.25 cMpc, the radius of the cylinder is 3.5  cMpc. Defined as the same as Nastasha[1].
- The recently heated particles are excluded. 
- Note: I took halos in snapshot z=0 while Nastasha took halos in snapshot z=0.1.
sbatch cal_halo_xraysb.sh

Using this data can generate [xray sb profile (sb vs halo radius)] or [xray sum sb vs halo masses]

### STEP2: xray sb halo profile
1. I divided the cylinder into azimuthal cylinder slices (radial bin) along the line of sight (along z axis).  In each slice I sum up the Xray luminosities of particles. 
2. In every radial bin I take the mean of all the halos in this bin to make mean profile. 
3. In every radial bin I take the median of halos whose particle counts in this bin >80. And I cut off the median profile if the bin has <90% halos whose have particle counts >80.  
sbatch cal_halo_xraysb_profile.sh (need parallel since it takes resources to filter particles in bins)
python3 plot_xray_radial_profile.py


### STEP3: xray sum lum vs halo masses
Right approach:
Take sum luminosity of particles of halo in r200c, divide by the same luminosity distance for every halo, which is $SB_{halo} = L{halo}/4 \pi d_L$ , where $d_L  = 1.48e27 cm$.