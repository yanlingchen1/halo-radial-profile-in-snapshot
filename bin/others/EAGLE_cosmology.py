from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy
import astropy.units as u
from unyt import arcmin, kpc
m_nu = [0.02, 0.02, 0.02] * u.eV
DESyr3 = FlatLambdaCDM(H0=67.8, Om0=0.307, m_nu=m_nu, Ob0=0.04825, Tcmb0=2.725)
print(f'{1/DESyr3.angular_diameter_distance(0.1)} 1 / Mpc') #
print(f'{1/DESyr3.arcsec_per_kpc_proper(0.1)} pkpc / arcsec') #
print(f'{1/DESyr3.arcsec_per_kpc_comoving(0.1)} ckpc / arcsec') #

print(f'{(1/DESyr3.arcsec_per_kpc_proper(0.1).value)**2} pkpc^2 / arcsec^2') #
print(f'{(1/DESyr3.arcsec_per_kpc_proper(0.1).value)**2 * 3600} pkpc^2 / arcmin^2') #
print(f'{(1/DESyr3.arcsec_per_kpc_proper(0.1).value)**2 * 3600 * 10} pkpc^2 / 10arcmin^2') #

print((204)**2*3.14)

print()