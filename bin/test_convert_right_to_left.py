import numpy as np

def convert_ri_to_le(y):
    return np.log10(np.power(10,y)/1e4/1e5/(1/360*(np.pi/180)**2))
print(convert_ri_to_le(0))
print(convert_ri_to_le(1))
print(convert_ri_to_le(2))
print(convert_ri_to_le(3))
print(convert_ri_to_le(4))
print(convert_ri_to_le(5))
