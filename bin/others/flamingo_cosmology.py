from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy
import astropy.units as u
from unyt import arcmin, kpc
m_nu = [0.02, 0.02, 0.02] * u.eV
DESyr3 = FlatLambdaCDM(H0=68.1, Om0=0.3046, m_nu=m_nu, Ob0=0.0486, Tcmb0=2.725)
print(1/DESyr3.angular_diameter_distance(0.1))
print(1/DESyr3.arcsec_per_kpc_proper(0.1))
print(1/DESyr3.arcsec_per_kpc_comoving(0.1))

print((1/DESyr3.arcsec_per_kpc_proper(0.1).value)**2*3600) #arcmin2/kpc2

print((1/DESyr3.arcsec_per_kpc_proper(0.1).value)*60*10) #10arcmin/kpc

print(((1/DESyr3.arcsec_per_kpc_proper(0.1).value)*60*10)**2/10) # 10arcmin2/kpc2

print((204*1.1)**2*3.14)

print()