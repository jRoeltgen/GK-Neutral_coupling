import numpy as np
import gkeyllIO
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

g = gkeyllIO.gkeyll('./test_data/gkeyll_data/', 'step23', True)
g.read_geometry()
g.read_data(78)
g.read_coeffs(78)
ptb=np.load('gkeyllGeometry/stored_data/gkeyll_b2_coordmapping.npy')
ptb_surfr=np.load('gkeyllGeometry/stored_data/gkeyll_b2_coordmapping_radialfaces.npy')
ptb_surfz=np.load('gkeyllGeometry/stored_data/gkeyll_b2_coordmapping_parallelfaces.npy')

g.interpolate_data(ptb)
g.interpolate_surfr_data(ptb_surfr)
g.interpolate_surfz_data(ptb_surfz)

g.calc_derived_data()
g.calc_derived_surfr_data()
g.calc_derived_surfz_data()

g.interpolate_data(ptb)
g.calc_derived_data()

import B2IO as b2
b2dat = b2.B2("./test_data/b2_data/")
sR = b2dat.gmtry["crx"].mean(axis=2).flatten()
sZ = b2dat.gmtry["cry"].mean(axis=2).flatten()

#Plot SOLPS interpolated data
#svals=g.interpolated_data["ti"].flatten()
svals=g.interpolated_data["ti"].flatten()
svals[svals<1000]=1000
norm=mpl.colors.LogNorm(vmin=svals.min(), vmax=svals.max())
fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (5,9))
markersize = 1.0
snorm=mpl.colors.LogNorm(vmin=svals.min(), vmax=svals.max())
#sim = ax.scatter(sR,sZ,c=svals,cmap='inferno',s=markersize, norm=snorm)
sim = ax.scatter(sR,sZ,c=svals,cmap='inferno',s=markersize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.0)
cbar = fig.colorbar(sim, cax=cax, orientation='vertical')
cbar.set_label(r'$n$', rotation=270, labelpad=25, fontsize=16)
ax.set_title('SOLPS')
ax.set_xlabel('R [m]')
ax.set_ylabel('Z [m]')
ax.axis("tight")
fig.tight_layout()

#Plot Gkeyll data directly
gvals = g.plot_data("ionTemp")

