import time
now = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"This notebook was generated at {now} ")

import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import warnings

from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.wcs import WCS
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

from astroquery.jplhorizons import Horizons

from photutils.aperture import CircularAperture as CAp
from photutils.aperture import CircularAnnulus as CAn
from photutils.centroids import centroid_com
from photutils import aperture_photometry as apphot
from photutils.detection import DAOStarFinder
from photutils.psf.groupstars import DAOGroup

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import patches
from matplotlib import gridspec

import ysvisutilpy as yvu
import ysfitsutilpy as yfu
import ysphotutilpy as ypu

warnings.filterwarnings('ignore', append=True, category=UserWarning)

DATAPATH = Path('C:/SNU/3-1/AO1/data')

vv = %version_information astropy, numpy, scipy, matplotlib, astroquery, ccdproc, photutils, ysfitsutilpy, ysphotutilpy, version_information

for i, pkg in enumerate(vv.packages):
    print(f"{i} {pkg[0]:10s} {pkg[1]:s}")
    
plt.style.use('default')
rcParams.update({'font.size':12})

###############################################################################

fpath = DATAPATH / 'M49_R.fits'

hdul = fits.open(fpath)
hdr = hdul[0].header
data = hdul[0].data
position, size = np.array([[1540, 1040], [900, 900]])
cut_hdu = Cutout2D(data=data, position=position, size=size)

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133, projection='3d')
yvu.zimshow(ax1, data)
yvu.zimshow(ax2, cut_hdu.data)

n, m = cut_hdu.data.shape
x, y = np.mgrid[:n, :m]
ax3.plot_surface(x, y, cut_hdu.data)

# Create a Rectangle patch
rect = patches.Rectangle(position-size/2, *size, lw=1, ec='r', fc='none')
# Add the patch to the Axes
ax1.add_patch(rect)

ax1.set_title(fpath.name)
ax2.set_title('Cutout Image')
ax3.set_title('3D projection')
plt.tight_layout()

###############################################################################

avg, med, std = sigma_clipped_stats(cut_hdu.data)  # by default, 3-sigma 5-iteration.
thresh_3sig = med + 3 * std
mask_3sig = (cut_hdu.data < thresh_3sig)
center = centroid_com(data=cut_hdu.data, mask=mask_3sig)
print(center)

fig, axs = plt.subplots(1, 1, figsize=(5, 6))
yvu.zimshow(axs, mask_3sig.astype(int))
yvu.zimshow(axs, cut_hdu.data, alpha=0.4)
axs.plot(*center, 'rx')
plt.tight_layout()

###############################################################################

# fitting signal with 2D Gaussian function to obtain rough measurement of FWHM 
n, m = cut_hdu.data.shape
x, y = np.mgrid[:n, :m]
const_init = models.Const2D(amplitude=med)
g_init = models.Gaussian2D(amplitude=cut_hdu.data.max(), x_mean=n/2, y_mean=m/2,
                            x_stddev=10., y_stddev=10.)
f_init = const_init + g_init
fit_f = fitting.LevMarLSQFitter()
f = fit_f(f_init, x, y, cut_hdu.data)
x_fwhm = f.x_stddev_1.value*gaussian_sigma_to_fwhm
y_fwhm = f.y_stddev_1.value*gaussian_sigma_to_fwhm

print(f'x_stddev : {f.x_stddev_1.value}', f'y_stddev : {f.y_stddev_1.value}')
print(f'conversion factor : {gaussian_sigma_to_fwhm}')
print(f'x_FWHM : {x_fwhm}', f'y_FWHM : {y_fwhm}')

fwhm = (x_fwhm + y_fwhm)/2
print(f'FWHM : {fwhm}')

###############################################################################

fwhm = 20
r_ap = 2 * fwhm
r_in = 4 * fwhm
r_out = 6 * fwhm
ap = CAp(positions=center, r=r_ap)
an = CAn(positions=center, r_in=r_in, r_out=r_out)

fig, axs = plt.subplots(1, 1, figsize=(5, 6))
yvu.zimshow(axs, cut_hdu.data)
ap.plot(axs, color='r', lw=2)
an.plot(axs, color='m', lw=2)
axs.plot(*center, 'rx')
plt.tight_layout()

###############################################################################

sky_mask = an.to_mask(method='center')
try:  # prior to photutils 0.7
    sky_vals = sky_mask[0].multiply(cut_hdu.data)
except TypeError:
    sky_vals = sky_mask.multiply(cut_hdu.data)
    
sky_vals = sky_vals[sky_vals > 0]
avg, med, std = sigma_clipped_stats(sky_vals, sigma=3, maxiters=10, std_ddof=1)

if med - avg < 0.3 * std:
    msky = med
else:
    msky = 2.5 * med - 1.5 * avg

print(f"Sky estimation: {msky:.3f} +- {std:.3f}")
    
fig, axs = plt.subplots(1, 1, figsize=(4, 3))

axs.hist(sky_vals, 100, histtype='step')
axs.axvline(msky, ls=':', color='r')
plt.tight_layout()

###############################################################################

phot = apphot(data=cut_hdu.data, apertures=ap)
phot["sky"] = msky
try:  # prior to photutils 0.7
    phot["source_sum"] = phot["aperture_sum"] - ap.area() * phot["sky"]
except TypeError:
    phot["source_sum"] = phot["aperture_sum"] - ap.area * phot["sky"]
    
phot["inst_mag"] = -2.5 * np.log10(phot["source_sum"] / hdr["EXPTIME"])
print(phot)

###############################################################################