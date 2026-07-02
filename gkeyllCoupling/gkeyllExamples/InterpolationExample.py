import numpy as np
import gkeyllIO
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import B2IO as b2
import eireneIO
import sys

frame = int(sys.argv[1])
gpath = "./"
g = gkeyllIO.gkeyll(gpath, 'hstep26', True, extra_species = ["molecule"])
g.read_geometry()
g.read_data(frame)
g.read_coeffs(frame)
geopath = "./python_data/hstep26_geodata/"
ptb=np.load(geopath+'gkeyll_b2_coordmapping.npy')
ptb_surfr=np.load(geopath+'gkeyll_b2_coordmapping_radialfaces.npy')
ptb_surfz=np.load(geopath+'gkeyll_b2_coordmapping_parallelfaces.npy')

edat = eireneIO.eirene("./step_full_device_D+D2_walled_off/gkeyll_w_2D2/")
#edat.load_extra_forts("./step_full_device_D+D2/gkeyll_w_2D2/")
edat.triangle_mesh.calc_incenter()
eR = edat.triangle_mesh.incenter[:,0]
eZ = edat.triangle_mesh.incenter[:,1]

b2dat = b2.B2("./step_full_device_D+D2_walled_off/baserun/")
sR = b2dat.gmtry["crx"].mean(axis=2).flatten()
sZ = b2dat.gmtry["cry"].mean(axis=2).flatten()

g.interpolate_data(ptb)
g.interpolate_surfr_data(ptb_surfr)
g.interpolate_surfz_data(ptb_surfz,)

g.calc_derived_data(b2dat, edat)
g.calc_derived_surfr_data(b2dat, edat)
g.calc_derived_surfz_data(b2dat, edat)

#Plot SOLPS interpolated data
sidx=1
svals=g.interpolated_data["ua"][:,:,sidx].flatten()
#svals[svals<1000]=1000
#norm=mpl.colors.LogNorm(vmin=1e16, vmax=svals.max())
norm=mpl.colors.SymLogNorm(vmin=svals.min(), vmax=svals.max(), linthresh=1)
#norm=mpl.colors.Normalize(vmin=svals.min(), vmax=svals.max())
#norm=mpl.colors.Normalize(vmin=-500, vmax=500)
fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (5,9))
markersize = 3.0
sim = ax.scatter(sR,sZ,c=svals,cmap='inferno',s=markersize, norm = norm)
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
#gvals = g.plot_data("ionM1", norm)

efig, eax = plt.subplots(nrows=1,ncols=1, figsize = (5,9))
evals =edat.fort31["ua"][:,:,sidx].flatten()
sim = eax.scatter(sR,sZ,c=evals,cmap='inferno',s=markersize, norm = norm)
divider = make_axes_locatable(eax)
cax = divider.append_axes('right', size='5%', pad=0.0)
cbar = fig.colorbar(sim, cax=cax, orientation='vertical')
cbar.set_label(r'$n_i$', rotation=270, labelpad=25, fontsize=16)
eax.set_title('EIRENE')
eax.set_xlabel('R [m]')
eax.set_ylabel('Z [m]')
eax.axis("tight")
efig.tight_layout()
