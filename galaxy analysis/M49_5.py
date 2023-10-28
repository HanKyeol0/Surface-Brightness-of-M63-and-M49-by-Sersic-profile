import time
now = time.strftime("%Y-%m-%d %H:%M:%S (GMT%z)")
print(f"This notebook was generated at {now} ")

vv = %version_information astropy, numpy, scipy, matplotlib, astroquery, ccdproc, photutils, ysfitsutilpy, ysphotutilpy, version_information
for i, pkg in enumerate(vv.packages):
    print(f"{i} {pkg[0]:10s} {pkg[1]:s}")
    
from pathlib import Path
import numpy as np

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astropy.stats import SigmaClip
from astroquery.sdss import SDSS

from matplotlib import pyplot as plt
from matplotlib import rcParams

from photutils.isophote import EllipseGeometry, Ellipse
from photutils.isophote import build_ellipse_model
from photutils.aperture import SkyCircularAperture as SCAp
from photutils.aperture import SkyCircularAnnulus as SCAn
from photutils.aperture import aperture_photometry, ApertureStats

from scipy.optimize import curve_fit

rcParams.update({'font.size':12})

DATADIR = Path('C:/SNU/3-1/AO1/data/')
NAME = 'M49'
imglist = list(DATADIR.glob('M49_R.fits'))
imglist.sort()
print(imglist)

interval = ZScaleInterval() # for zscale image plot

###############################################################################

# Sersic law (magnitude)
def sersic(r, mu_eff, r_eff, n):
    bn = 2 * n - 1/3
    return mu_eff + (2.5 * bn / np.log(10)) * ((r / r_eff)**(1/n) - 1)


# Sersic law (flux)
def sersic_flux(r, Ie, re, n):
    return Ie*np.exp(-(2*n-1/3)*((r/re)**(1./n)-1))


# Double components of Sersic law (mag)
def double_sersic(r, Ie1, re1, n1, Ie2, re2, n2):
    I = sersic_flux(r, Ie1, re1, n1) + sersic_flux(r, Ie2, re2, n2)
    return 22.5 - 2.5*np.log10(I)  # SDSS zero-point; you may have to change it 


class SloanGalaxySurfacePhot:
    def __init__(self, imgpath, sky_r0, pixel_scale, name='',
                 band='', mag0=22.5):
        '''
        This class loads SDSS fits file of an image of a galaxy, queries stars
        in the field, does aperture photometry for these stars to obtain FWHM
        of the PSF, fit elliptical isophote to this galaxy, and carries out
        surface photometry.

        Parameters
        ----------
        imgpath : pathlib Path
            Path of subject SDSS fits image
        sky_r0 : int
            The outer boundary for sky estimation in pixel
        pixel_scale : float
            pixel scale in arcsec
        name : str, optional
            The name of the target in the input image. The default is ''.
        band : str, optional
            The filter used for the input image. The default is ''.
        mag0 : float, optional
            The zero-point for calculating magnitudes. The default is 22.5,
            which is the standard for the calibrated SDSS images.

        Returns
        -------
        None.

        '''
        self.imgpath = imgpath
        self.sky_r0 = sky_r0
        self.pixel_scale = pixel_scale
        self.mag0 = mag0
        self.name = name
        self.band = band
        
    def set_image(self, ext=0, display=True, vmax=None):
        self.img, self.hdr = fits.getdata(self.imgpath, header=True, ext=ext)
        self.wcs = WCS(self.hdr)
        self.y_center, self.x_center = np.array(self.img.shape)/2
        
        r0 = self.sky_r0   # outer boundary for sky estimation

        # --- Background estimation for determining backgroun level --- #

        x1d = np.arange(0, self.img.shape[1], 1)
        y1d = np.arange(0, self.img.shape[0], 1)
        xx, yy = np.meshgrid(x1d, y1d, sparse=True)
        z = ((xx-self.x_center)**2.0 + (yy-self.y_center)**2.0 - r0**2.0)
        self.sky_region = (z > 0.0)

        avg, med, std = sigma_clipped_stats(self.img[self.sky_region],
                                            sigma=3.0)
        self.sky_val, self.sky_sig = 3.0*med - 2.0*avg, std
        print("Image: "+self.imgpath.stem)
        print("sky level: {0:.4f}".format(self.sky_val))
        print("sky sigma: {0:.4f}".format(self.sky_sig))
        # ---------- #
        if display:
            fig, ax = plt.subplots(figsize=(5, 5))
            vmin, _vmax = interval.get_limits(self.img)
            if vmax:
                vmax = vmax if vmax > _vmax else _vmax
            else:
                vmax = _vmax
            vlimits = {'vmin': vmin, 'vmax': vmax}
            ax.imshow(self.img, origin='lower', **vlimits)
            ax.imshow(self.sky_region, origin='lower', alpha=0.2, cmap='gray')
            plt.title(f'{self.name} ({self.band}-band)')
        
    def set_kwargs_simple(self):
        kwargs = {"x0":self.x_center-1, "y0":self.y_center-1, 'pa':np.pi/2,
                  "eps":0.05, "sma":50.0, "linear_growth":False,
                  "astep":0.1, "fix_center":False, "fix_eps":False, "fix_pa":False}
        self.kwargs = kwargs
        
    def set_kwargs(self, kwargs):
        self.kwargs = kwargs
        
    def ellipse(self, display=True, backgr=True, minsma=0.1, maxsma=100.,
                nclip=2, vmax=None, **kwargs):
        print('Start: ' + time.ctime())
        start_time = time.time()

        # running photutils/ellipse task
        g = EllipseGeometry(**self.kwargs)
        ellipse = Ellipse(self.img, geometry=g)
        
        self.backgr = backgr
        
        isolist = ellipse.fit_image(sma0=self.kwargs['sma'],
                                    step=self.kwargs['astep'], nclip=nclip,
                                    minsma=minsma, maxsma=maxsma,
                                    maxrit=maxsma, **kwargs)

        end_time = time.time()
        print(f'Elapsed time : {(end_time - start_time)/60:.2f} min '+
              f'({end_time - start_time:.2f} sec)')
        self.isolist = isolist
        
        basedir = self.imgpath.parent
        imgname = self.imgpath.stem                   # image name
        self.mod_path = basedir/(imgname+'_mod.fits') # output model path
        self.res_path = basedir/(imgname+'_res.fits') # output resid path
        
        if self.mod_path.exists() and self.res_path.exists():
            self.mod_path.unlink()
            self.res_path.unlink()
        
        if display:
            self.display_omr(save=True, vmax=vmax)
        
    def display_omr(self, vmax=None, save=True):
        if self.mod_path.exists() and self.res_path.exists():
            model = fits.getdata(self.mod_path)
            residual = fits.getdata(self.res_path)
        else:
            # Making model, residual images
            start_time = time.time()
            backgr_level = self.sky_val if self.backgr else 0
            model = build_ellipse_model(self.img.shape, self.isolist,
                                        fill=backgr_level)
            residual = self.img - model
            end_time = time.time()
            print('Elapsed time (Making model image) : ' +
                  f'{(end_time - start_time)/60:.2f} min ' +
                  f'({end_time - start_time:.2f} sec)')        
            if save:
                fits.writeto(self.mod_path, model.astype('float32'),
                             overwrite=True)
                fits.writeto(self.res_path, residual.astype('float32'),
                             overwrite=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 5), nrows=1, ncols=3)
        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
        vmin, _vmax = interval.get_limits(self.img)
        if vmax:
            vmax = vmax if vmax > _vmax else _vmax
        else:
            vmax = _vmax
        vlimits = {'vmin': vmin, 'vmax': vmax}
        ax1.imshow(self.img, origin='lower', **vlimits)
        ax1.set_title('Data')

        smas = np.linspace(100, 500, 5)
        for sma in smas:
            iso = self.isolist.get_closest(sma)
            x, y, = iso.sampled_coordinates()
            ax1.plot(x, y, color='g', lw=1)

        ax2.imshow(model, origin='lower', **vlimits)
        ax2.set_title('Ellipse Model')

        ax3.imshow(residual, origin='lower', **vlimits)
        ax3.set_title('Residual')
        ax3.text(0.95, 0.05, f'{self.name} ({self.band}-band)', ha='right',
                 transform=ax3.transAxes, c='w', fontweight='bold')
        return ax1, ax2, ax3
        
    def set_ref_stars(self, maglimit=16, display=True, vmax=None):
        bnds = self.wcs.pixel_to_world([0,self.wcs.array_shape[0]],
                                       [0,self.wcs.array_shape[1]])
        ra_bnds = np.sort(bnds.ra.value)
        dec_bnds = np.sort(bnds.dec.value)
        # SQL query for SDSS
        # (https://skyserver.sdss.org/dr12/en/help/docs/realquery.aspx)
        self.sql_query = f'''
            SELECT u,g,r,i,z,ra,dec, flags_r
            FROM Star
            WHERE
            ra BETWEEN {ra_bnds[0]} and {ra_bnds[1]}
            AND dec BETWEEN {dec_bnds[0]} and {dec_bnds[1]}
            AND ((flags_r & 0x10000000) != 0)
            -- detected in BINNED1
            AND ((flags_r & 0x8100000c00a4) = 0)
            -- not EDGE, NOPROFILE, PEAKCENTER, NOTCHECKED, PSF_FLUX_INTERP,
            -- SATURATED, or BAD_COUNTS_ERROR
            AND (((flags_r & 0x400000000000) = 0) or (psfmagerr_r <= 0.2))
            -- not DEBLEND_NOPEAK or small PSF error
            -- (substitute psfmagerr in other band as appropriate)
            AND (((flags_r & 0x100000000000) = 0) or (flags_r & 0x1000) = 0)
            -- not INTERP_CENTER or not COSMIC_RAY
            AND g < {maglimit}
            AND r < {maglimit}
            '''
        self.star_table = SDSS.query_sql(self.sql_query, timeout=9999)
        self.co_star = SkyCoord(ra=self.star_table['ra'],
                               dec=self.star_table['dec'], unit='deg')
        
        if display:
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111, projection=self.wcs)
            vmin, _vmax = interval.get_limits(self.img)
            if vmax:
                vmax = vmax if vmax > _vmax else _vmax
            else:
                vmax = _vmax
            vlimits = {'vmin': vmin, 'vmax': vmax}
            ax.imshow(self.img, origin='lower', **vlimits)
            ax.scatter(self.co_star.ra.value, self.co_star.dec.value, ec='r',
                       fc='none', transform=ax.get_transform('world'))
        
    def apphot_stars(self, display=True, vmax=0.7):
        # initial apertures to obtain fwhm
        aperture_init = SCAp(self.co_star, r=4. * u.arcsec)
        aperstats = ApertureStats(self.img, aperture_init, wcs=self.wcs)
        
        fwhm = aperstats.fwhm.mean() / u.pix * u.arcsec * self.pixel_scale
        r_ap = 2 * fwhm
        r_in = 4 * fwhm
        r_out = 6 * fwhm
        self.fwhm = fwhm
        
        # define new apertures with derived fwhm
        aperture = SCAp(self.co_star, r=r_ap)
        annulus = SCAn(self.co_star, r_in=r_in, r_out=r_out)
        
        # aperture photometry
        phot_table = aperture_photometry(self.img, aperture, wcs=self.wcs)
        
        # re-define apertures in pixel scale
        ap = aperture.to_pixel(self.wcs)
        an = annulus.to_pixel(self.wcs)
        
        # background estimation with annuli
        sigclip = SigmaClip(sigma=3.0, maxiters=10)
        anstats = ApertureStats(self.img, an, sigma_clip=sigclip)
        med = anstats.median
        avg = anstats.mean
        std = anstats.std
        
        msky = np.zeros_like(med)
        mode_mask = (med - avg)/std < 0.3
        msky[mode_mask] = med[mode_mask]
        msky[~mode_mask] = (2.5*med - 1.5*avg)[~mode_mask]
        #self.mu_sky = self.mag0 -2.5*np.log10(np.mean(msky))
        
        # sky background subtraction
        ap_area = ap.area_overlap(self.img)
        total_sky = msky * ap_area
        phot_skysub = phot_table['aperture_sum'] - total_sky
        phot_table['total_sky'] = total_sky
        phot_table['aperture_sum_skysub'] = phot_skysub
        
        self.phot_table = phot_table
        
        if display:
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111, projection=self.wcs)
            vmin, _vmax = interval.get_limits(self.img)
            if vmax:
                vmax = vmax if vmax > _vmax else _vmax
            else:
                vmax = _vmax
            vlimits = {'vmin': vmin, 'vmax': vmax}
            ax.imshow(self.img, origin='lower', **vlimits)
            ap.plot(ax, color='r', lw=0.3)
            an.plot(ax, color='w', lw=0.3)
            ax.set_xlabel('RA')
            ax.set_ylabel('DEC')
            # # inset axes....
            # axins = ax.inset_axes([0.65, 0.20, 0.30, 0.30])
            # axins.imshow(self.img, origin="lower", **vlimits)
            # ap.plot(axins, color='r', lw=0.3)
            # an.plot(axins, color='w', lw=0.3)
            # # sub region of the original image
            # x1, x2, y1, y2 = 1892,1983,189,280
            # axins.set_xlim(x1, x2)
            # axins.set_ylim(y1, y2)
            # axins.set_xticklabels([])
            # axins.set_yticklabels([])
            # ax.indicate_inset_zoom(axins, edgecolor="black")
            # plt.tight_layout()
        
    def get_mag0(self):
        mag = self.star_table[self.band]
        inst_mag = -2.5*np.log10(self.phot_table['aperture_sum_skysub'])
        mag0 = mag - inst_mag
        return mag0
        
    def flux2mag(self, flux):
        return self.mag0 -2.5*np.log10(flux)
    
    def set_radial_profile(self, display=True):
        # Unit conversion
        self.r_sma = self.isolist.sma * self.pixel_scale    # pixel to arcsec
        self.mu = (self.flux2mag(self.isolist.intens)
                 - 2.5*np.log10(self.pixel_scale**(-2))) # mag/pix^2 to mag/"^2
        flux_err = np.sqrt(self.isolist.int_err**2 + self.sky_sig**2)
        self.mu_err = 2.5*flux_err/self.isolist.intens/np.log(10)
        if display:
            self.display_radial_profile()
        
    def display_radial_profile(self):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(self.r_sma**0.25, self.mu, c='w', zorder=-100)
        ylim = ax.get_ylim()
        ax.errorbar(self.r_sma**0.25, self.mu, yerr=self.mu_err,
                    ms=3.0, c='k', alpha=0.6)
#       ax.axhline(self.mu_sky - 2.5*np.log10(self.pixel_scale**(-2)),
#                    linestyle='--', color='gray', lw=1)
        ax.axvline(self.fwhm.value**0.25,
                   linestyle='--', color='gray', lw=1)
        
        if hasattr(self, 'popt'):
            ax.plot(self.r_sma**0.4, self.fitfunc(self.r_sma, *self.popt),
                    c='r')
            ax.text(0.95, 0.95,
                    f'RMSE = {self.rmse:.3f}',
                    va='top', ha='right', transform=ax.transAxes)
            ax.text(0.95, 0.90,
                    f'n = {self.popt[2]:.3f} $\pm$ {self.perr[2]:.3f}',
                    va='top', ha='right', transform=ax.transAxes)
            if self.fitfunc.__name__ == 'double_sersic':
                ax.plot(self.r_sma**0.25,
                        self.flux2mag(sersic_flux(self.r_sma, *self.popt[:3])),
                        c='orange')
                ax.plot(self.r_sma**0.25,
                        self.flux2mag(sersic_flux(self.r_sma, *self.popt[3:])),
                        c='dodgerblue')
                ax.text(0.95, 0.85,
                        f'n = {self.popt[5]:.3f} $\pm$ {self.perr[5]:.3f}',
                        va='top', ha='right', transform=ax.transAxes)
                    
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        secax = ax.secondary_xaxis('top', functions=(lambda x: x**4,
                                                     lambda x: x**0.25))
        secax.set_xlabel('$a$ (semi-major axis) [arcsec]')
        secax.grid(True, which='minor')
        ax.set_xlabel("$a^{1/4}$ [arcsec$^{1/4}$]")
        ax.set_ylabel(f"$\mu_{self.band}$ "+r"[mag ${\rm arcsec^{-2}}$]")
        plt.title(f'{self.name} ({self.band}-band)')
        plt.tight_layout()
        return ax
    
    def fit_radial_profile(self, mask=None, double=False):
        if mask is None:
            mask = self.r_sma > self.fwhm.value
        else:
            mask = np.logical_and(self.r_sma > self.fwhm.value, mask)
        
        if False in np.isfinite(self.mu):
            mask = np.logical_and(mask, np.isfinite(self.mu))
            
        if double:
            func = double_sersic
            idx1 = int(len(self.mu)/1.3)
            idx2 = int(len(self.mu)/1.1)
            p0 = [self.isolist.intens[idx1]+3, self.r_sma[idx1], 4.,
                  self.isolist.intens[idx2]+3, self.r_sma[idx2], 1.]
            bounds = ([0.0, 10., 0.01, 0.0, 10., 0.01],
                      [10., 200., 10., 10., 200., 3.])
        else:
            func = sersic
            idx = int(len(self.mu)/1.3) # ! hard-coded.
            p0 = [self.mu[idx]+3, self.r_sma[idx], 4.]
            bounds=([15., 10., 0.01], [30., 200., 10.])
        popt, pcov = curve_fit(func, self.r_sma[mask], self.mu[mask], p0=p0,
                               sigma=self.mu_err[mask], bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        self.rmse = np.std(self.mu[mask] - func(self.r_sma[mask], *popt))
        self.popt, self.perr, self.fitfunc = popt, perr, func
        
        
def display_color_profile(obj_band1, obj_band2, title):
    color = obj_band1.mu - obj_band2.mu
    color_err = np.sqrt(obj_band1.mu_err**2 + obj_band2.mu_err**2)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(obj_band1.r_sma**0.25, color, c='w')
    ylim = ax.get_ylim()
    ax.errorbar(obj_band1.r_sma**0.25, color, yerr=color_err,
                ms=3.0, c='k', alpha=0.6)
    ax.axvline(obj_band1.fwhm.value**0.25,
               linestyle='--', color='gray', lw=0.5)
    ax.set_ylim(ylim)
    
    # ax.invert_yaxis()
    secax = ax.secondary_xaxis('top', functions=(lambda x: x**4,
                                                 lambda x: x**0.25))
    secax.set_xlabel('$a$ (semi-major axis) [arcsec]')
    ax.set_xlabel("$a^{1/4}$ [arcsec$^{1/4}$]")
    ax.set_ylabel(f"$\mu_{obj_band1.band} - \mu_{obj_band2.band}$"+
                  r" [mag ${\rm arcsec^{-2}}$]")
    plt.title(title)
    plt.tight_layout()


def display_profile_prop(prop, obj_band1, obj_band2, title,
                         savedir=None):
    fig, ax = plt.subplots(figsize=(6,4))
    if prop == 'eps':
        y1, y2 = obj_band1.isolist.eps, obj_band2.isolist.eps
        y1_err = obj_band1.isolist.ellip_err
        y2_err = obj_band2.isolist.ellip_err
    else:
        y1 = getattr(obj_band1.isolist, prop)
        y2 = getattr(obj_band2.isolist, prop)
        y1_err = getattr(obj_band1.isolist, prop+'_err')
        y2_err = getattr(obj_band2.isolist, prop+'_err')
    
    ax.plot(obj_band1.r_sma**0.25, y1, c='w', label=None)
    ax.plot(obj_band2.r_sma**0.25, y2, c='w', label=None)
    ylim = ax.get_ylim()

    ax.errorbar(obj_band1.r_sma**0.25, y1, yerr=y1_err,
                ms=3.0, c='b', alpha=0.6, label=obj_band1.band+'-band')
    ax.errorbar(obj_band2.r_sma**0.25, y2, yerr=y2_err,
                ms=3.0, c='r', alpha=0.6, label=obj_band2.band+'-band')
    ax.axvline(obj_band1.fwhm.value**0.25,
               linestyle='--', color='gray', lw=0.5)
    ax.set_ylim(ylim)
    # ax.invert_yaxis()
    secax = ax.secondary_xaxis('top', functions=(lambda x: x**4,
                                                 lambda x: x**0.25))
    secax.set_xlabel('$a$ (semi-major axis) [arcsec]')
    ax.set_xlabel("$a^{1/4}$ [arcsec$^{1/4}$]")
    ax.set_ylabel(prop)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if savedir:
        plt.savefig(savedir/f'{title.lower()}_{prop}.png',
                    overwrite=True, dpi=300)
    
###############################################################################

# SDSS pixel scale

pixel_scale = 0.388

print('pixel_scale : ', pixel_scale)

imgpath = imglist[0]

m89_g = SloanGalaxySurfacePhot(imgpath, sky_r0=700, pixel_scale=pixel_scale,
                               name='M49', band='r', mag0=22.5)

# fetch image
m89_g.set_image()

# query stars from SDSS to check the zeropoint
m89_g.set_ref_stars(maglimit=18.5, display=True)

m89_g.apphot_stars()

mag0_from_apphot_g = m89_g.get_mag0()

print(mag0_from_apphot_g)

print(np.mean(mag0_from_apphot_g))

print(m89_g.fwhm)

m89_g.set_kwargs_simple()  # set keyword arguments with default setting
                           # use set_kwargs function for custom kwargs.
m89_g.ellipse(minsma=0.1, maxsma=700., vmax=0.7)

m89_g.set_radial_profile()

m89_g.fit_radial_profile(double=False)
m89_g.display_radial_profile()

for sgsp in [m89_g]:
    sgsp.display_radial_profile()
    
    # color profile
display_color_profile(m89_g, 'M49')

for sgsp in [m89_g]:
    tflux = sgsp.isolist.tflux_e[-1]
    mag = sgsp.flux2mag(tflux)
    mag_err = 2.5*np.sqrt(tflux)/tflux/np.log(10)
    # npix_e = sgsp.isolist.npix_e[-1]
    # mag_err = 2.5*np.sqrt(tflux+npix_e*sgsp.sky_sig**2)/tflux/np.log(10)
    print(f'$ {sgsp.band} = {mag:.3f} \pm {mag_err:.3f} $')
