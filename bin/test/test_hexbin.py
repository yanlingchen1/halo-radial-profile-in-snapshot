import numpy as np
import matplotlib.pyplot as plt

# Example data
r = np.linspace(0, 10, 100)  # Radius values
nH = np.exp(-r)  # Density values
T = np.random.uniform(1, 10, size=len(r))  # Temperature values

# Create a hexbin plot with color coding
plt.hexbin(r, nH, C=T, cmap='coolwarm', gridsize=20)

# Add colorbar and labels
cbar = plt.colorbar()
cbar.set_label('Temperature (T)')

plt.xlabel('Radius (r)')
plt.ylabel('Density (nH)')
plt.title('Particle Profiles with Temperature Markings')

plt.savefig('tsthexbin.png')