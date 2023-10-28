import numpy as np
from pathlib import Path
import time
import warnings

from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.nddata import CCDData
import astroscrappy
import astroalign as aa
from ccdproc import combine
from matplotlib import pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings('ignore', append=True, category=UserWarning)

OBJNAME = 'M49'

DATADIR = Path('C:/SNU/3-1/AO1/data')
OBJDIR = DATADIR / OBJNAME

bias_list = list(DATADIR.glob('cal*bias.fit'))
dark_list_sci = list(DATADIR.glob('cal*dk180.fit'))
dark_list_flat = list(DATADIR.glob('cal*dk2.fit'))
flat_list_B = list(DATADIR.glob('skyflat*B.fit'))
flat_list_Ha = list(DATADIR.glob('skyflat*Ha.fit'))
flat_list_I = list(DATADIR.glob('skyflat*I.fit'))
flat_list_Oiii = list(DATADIR.glob('skyflat*Oiii.fit'))
flat_list_R = list(DATADIR.glob('skyflat*R.fit'))
flat_list_Sii = list(DATADIR.glob('skyflat*Sii.fit'))
flat_list_U = list(DATADIR.glob('skyflat*U.fit'))
flat_list_V = list(DATADIR.glob('skyflat*V.fit'))
##sci_list_B  = list(DATADIR.glob(OBJNAME + '*B.fit'))
sci_list_I  = list(DATADIR.glob(OBJNAME + '*I.fit'))
sci_list_R  = list(DATADIR.glob(OBJNAME + '*R.fit'))
##sci_list_V  = list(DATADIR.glob(OBJNAME + '*V.fit'))

print(f"Bias: {len(bias_list):d} frames")
print(f"Dark: {len(dark_list_sci):d} frames")
print(f"Dark: {len(dark_list_flat):d} frames")
print(f"Flat (sky; B-band): {len(flat_list_B):d} frames")
print(f"Flat (sky; R-band): {len(flat_list_R):d} frames")
print(f"Flat (sky; I-band): {len(flat_list_I):d} frames")
print(f"Flat (sky; U-band): {len(flat_list_U):d} frames")
print(f"Flat (sky; V-band): {len(flat_list_V):d} frames")
print(f"Science (object; R-band): {len(sci_list_R):d} frames")
print(f"Science (object; I-band): {len(sci_list_I):d} frames")

###############################################################################

fpath = sci_list_R[0]
data, hdr = fits.getdata(fpath, header=True)

gain = hdr['EGAIN']

for keys in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME']:
    print(keys+" = "+str(hdr[keys]))
    
###############################################################################
    
bias0 = fits.getdata(bias_list[0])
dark0 = fits.getdata(dark_list_sci[0])
flat0 = fits.getdata(flat_list_R[0])
sci0  = fits.getdata(sci_list_R[0])
label = ["Bias", "Dark", "Sky Flat", "Science Object (Raw)"]

interval = ZScaleInterval()

fig, axs = plt.subplots(2, 2, figsize=(9,6))
for i, img0 in enumerate([bias0, dark0, flat0, sci0]):
    vmin, vmax = interval.get_limits(img0)
    ax = axs[i // 2][i % 2]
    ax.imshow(img0, origin='lower', vmin=vmin, vmax=vmax)
    ax.tick_params(axis='both', length=0.0, labelleft=False, labelbottom=False)
    ax.text(0.05, 0.95, label[i], fontsize=15.0, fontweight='bold',
            transform=ax.transAxes, ha='left', va='top')
plt.tight_layout()

###############################################################################

bias1, bias_hdr = fits.getdata(bias_list[0], header=True)            
bias1 = np.array(bias1).astype('float64')

bias2 = fits.getdata(bias_list[0])            
bias2 = np.array(bias2).astype('float64')

dbias = bias2 - bias1

gain = bias_hdr['EGAIN']

RN = np.std(dbias)*gain / np.sqrt(2)
print('Readout Noise is {0:.2f}'.format(RN))

name = []
RN = []
for i in range(len(bias_list)-1):
    bias1 = fits.getdata(bias_list[i]).astype('float64')
    bias2 = fits.getdata(bias_list[i+1]).astype('float64')
    dbias = bias2 - bias1

    print(i,'st',np.std(dbias)*gain / np.sqrt(2))
    RN.append(np.std(dbias)*gain / np.sqrt(2))
print(np.mean(RN))    
rdnoise = np.mean(RN)

print(gain)

###############################################################################

OBJDIR = DATADIR / OBJNAME
RAWDIR = OBJDIR / 'raw'
OUTDIR = OBJDIR / 'reduced'

for i in np.arange(len(bias_list)):
    bias_hdr = fits.getheader(bias_list[i])
    print(f"\nBias frame {i+1:d}")
    for keys in ['DATE-OBS', 'EXPTIME']:
        print("  "+keys+" = "+str(bias_hdr[keys]))
        
###############################################################################

bias_array = np.empty((len(bias_list), bias0.shape[0], bias0.shape[1]))

for i in range(len(bias_list)):
    bias_data = fits.getdata(bias_list[i])
    bias_array[i, :, :] = bias_data

bias_med = np.median(bias_array, axis=0)
bias_stack = []

for i in range(len(bias_list)):
    bias_data, bias_hdr = fits.getdata(bias_list[i], header=True)
    bias = CCDData(data=bias_data, header=bias_hdr, unit='adu')
    bias_stack.append(bias)

mbias = combine(bias_stack, sigma_clip=True,
                sigma_clip_high_thresh=3, sigma_clip_low_thresh=3)

bias_hdr['NFRAMES'] = len(bias_list)
fits.writeto(DATADIR/"MBias.fits", mbias.data, bias_hdr, overwrite=True)

fig, ax = plt.subplots(1, 1, figsize=(5,3))
vmin, vmax = interval.get_limits(mbias)
ax.imshow(mbias, origin='lower', vmin=vmin, vmax=vmax)
ax.tick_params(axis='both', length=0.0, labelleft=False, labelbottom=False)
ax.text(0.50, 0.96, "Master Bias", fontsize=12.0, fontweight='bold',
        transform=ax.transAxes, ha='center', va='top')
ax.text(0.50, 0.88, "(combined with sigma_clipping)", fontsize=11.0,
        transform=ax.transAxes, ha='center', va='top')
ax.text(0.04, 0.12, f"Mean bias level: {np.mean(mbias):.1f}", fontsize=10.0,
        transform=ax.transAxes, ha='left', va='bottom')
ax.text(0.04, 0.04, f"Bias fluctuation: {np.std(mbias):.2f}", fontsize=10.0,
        transform=ax.transAxes, ha='left', va='bottom')
plt.tight_layout()

print(f'bias_med fluctuation : {np.std(bias_med):.2f}')
print(f'bias_sc fluctuation : {np.std(mbias):.2f}')

###############################################################################

def make_mdark(dark_list):
    for i in np.arange(len(dark_list)):
        dark_hdr = fits.getheader(dark_list[i])
        print(f"\nDark frame {i+1:d}")
        for keys in ['DATE-OBS', 'EXPTIME']:
            print("  "+keys+" = "+str(dark_hdr[keys]))

    dark_stack = []
    for i in range(len(dark_list)):
        dark_data, dark_hdr = fits.getdata(dark_list[i], header=True)
        dark_bn = (dark_data - mbias.data)# / dark_hdr['EXPTIME']
        dark = CCDData(data=dark_bn, header=dark_hdr, unit='adu')    
        dark_stack.append(dark)

    mdark = combine(dark_stack, sigma_clip=True,
                    sigma_clip_high_thresh=3, sigma_clip_low_thresh=3)

    mdark.data[mdark.data < 0.] = 0.

    dark_hdr['NFRAMES'] = len(dark_list)
    fits.writeto(DATADIR/f"MDark{dark_hdr['EXPTIME']:.0f}.fits",
                 mdark, dark_hdr, overwrite=True)

    fig, ax = plt.subplots(1, 1, figsize=(5,3))
    vmin, vmax = interval.get_limits(mdark)
    ax.imshow(mdark, origin='lower', vmin=-1, vmax=1)
    ax.tick_params(axis='both', length=0.0, labelleft=False, labelbottom=False)
    ax.text(0.50, 0.96, "Master Dark", fontsize=12.0, fontweight='bold',
            transform=ax.transAxes, ha='center', va='top')
    ax.text(0.50, 0.88, "(sc combined, bias-subtracted)", fontsize=11.0,
            transform=ax.transAxes, ha='center', va='top')
    ax.text(0.04, 0.12, f"Mean dark level: {np.mean(mdark):.1f} (count)",
            fontsize=10.0, transform=ax.transAxes, ha='left', va='bottom')
    ax.text(0.04, 0.04, f"Dark fluctuation: {np.std(mdark):.2f}", fontsize=10.0,
            transform=ax.transAxes, ha='left', va='bottom')
    plt.tight_layout()
    
    return mdark

mdark_sci, mdark_flat = make_mdark(dark_list_sci), make_mdark(dark_list_flat)

###############################################################################

mflat_list = []
for flat_list in [flat_list_R, flat_list_I]:
    for i in np.arange(len(flat_list)):
        flat_hdr = fits.getheader(flat_list[i])
        print(f"\nFlat frame {i+1:d}")
        for keys in ['DATE-OBS', 'EXPTIME', 'FILTER']:
            print("  "+keys+" = "+str(flat_hdr[keys]))

    flat_stack = []
    for i in np.arange(len(flat_list)):
        flat_data, flat_hdr = fits.getdata(flat_list[i], header=True)  
        flat_bd = (flat_data - mbias.data - mdark_flat.data)
        flat_bdn = flat_bd/flat_bd.max()
        flat_stack.append(CCDData(data=flat_bdn, unit='adu'))

    mflat = combine(flat_stack, sigma_clip=True,
                      sigma_clip_low_thresh=3,
                      sigma_clip_high_thresh=3)

    filter_now = flat_hdr['FILTER']
    flat_hdr['NFRAMES'] = len(flat_list)
    fits.writeto(DATADIR/f"MFlat{filter_now}.fits", mflat.data,
                 header=flat_hdr, overwrite=True)
    mflat_list.append(mflat.data)

    fig, ax = plt.subplots(1, 1, figsize=(5,3))
    vmin, vmax = interval.get_limits(mflat)
    ax.imshow(mflat, origin='lower', vmin=vmin, vmax=vmax)
    ax.tick_params(axis='both', length=0.0, labelleft=False, labelbottom=False)
    ax.text(0.50, 0.96, f"Master Flat ({filter_now})", fontsize=12.0,
            fontweight='bold', transform=ax.transAxes, ha='center', va='top')
    ax.text(0.50, 0.88, "(sc combined, bias/dark-subtracted)",
            fontsize=11.0, transform=ax.transAxes, ha='center', va='top')
    ax.text(0.04, 0.12, f"Flat sensitivity range: {100*mflat.data.min():.1f}"
            + f" - {100*mflat.data.max():.1f}%", ha='left', va='bottom',
            fontsize=10.0, color='w', transform=ax.transAxes)
    ax.text(0.04, 0.04, f"Flat fluctuation: {100*np.std(mflat.data):.2f}%",
            fontsize=10.0, color='w', transform=ax.transAxes,
            ha='left', va='bottom')
    plt.tight_layout()
    
###############################################################################

def crrej(data, header, readnoise, method='CR'):
    gain = header['EGAIN']

    LACOSMIC_KEYS = dict(sigclip=4.5, sigfrac=0.5, objlim=5.0,
                 satlevel=np.inf, niter=4, #pssl=0.0
                 cleantype='medmask', fsmode='median', psfmodel='gauss',
                 psffwhm=2.5, psfsize=7, psfk=None, psfbeta=4.765)

    if method == 'LA':
        ccd_LA = CCDData(data=[0], header=header, unit='adu')

        m_LA, ccd_LA.data = astroscrappy.detect_cosmics(
                                data,
                                sepmed=False,
                                gain=gain,
                                readnoise=readnoise,
                                **LACOSMIC_KEYS)
        return ccd_LA, m_LA
    elif method == 'CR':
        ccd_CR = CCDData(data=[0], header=header, unit='adu')

        m_CR, ccd_CR.data = astroscrappy.detect_cosmics(
                                data,
                                sepmed=True,
                                gain=gain, 
                                readnoise=readnoise,
                                **LACOSMIC_KEYS)
        return ccd_CR, m_CR
    else:
        raise ValueError('kwarg "method" should be either "LA" or "CR".')
    

def preproc(sci_list, mbias, mdark, mflat, rdnoise, show=True, save=True):
    for i in range(len(sci_list)):
        sci_path = sci_list[i]
        sci_data, sci_hdr  = fits.getdata(sci_path, header=True)
        sci_data0 = sci_data.astype('float')
        
        sci_data1 = sci_data0 - mbias
        sci_data1 -= mdark
        sci_data1 /= mflat

        sci_crrej, sci_mask = crrej(sci_data1, sci_hdr, rdnoise, method='CR')

        if show:
            fig, axs = plt.subplots(1, 3, figsize=(14,3))
            title = ["Raw Data",
                      "Preprocessed Data",
                      "Cosmic Ray Rejeccted Data"]
            for i, sci_data in enumerate([sci_data0, sci_data1, sci_crrej]):
                ax = axs[i]
                vmin, vmax = interval.get_limits(sci_data)
                ax.imshow(sci_data, cmap='viridis',
                          origin='lower', vmin=vmin, vmax=vmax)
                ax.tick_params(axis='both', length=0.0,
                               labelleft=False, labelbottom=False)
                if i == 0:
                    ax.text(0.04, 0.04, sci_path.name, fontsize=12.0, 
                            transform=ax.transAxes, ha='left', va='bottom')
                ax.text(0.50, 0.96, title[i], fontsize=12.0, fontweight='bold',
                        transform=ax.transAxes, ha='center', va='top')
            plt.tight_layout()

        now = time.strftime("%Y-%m-%d %H:%M:%S (GMT%z)")
        sci_hdr['RDNOISE'] = rdnoise
        sci_hdr['history'] = 'Preprocessed at ' + now

        scihdu = fits.PrimaryHDU(data=sci_crrej, header=sci_hdr)
        mskhdu = fits.ImageHDU(data=sci_mask.astype('int'), name='MASK')
        hdul = fits.HDUList([scihdu, mskhdu])
        hdul.writeto(DATADIR/('p'+sci_path.name+'s'), overwrite=True)
        print(f'Done: {sci_path.name}')


for sci_list, mflat in zip([sci_list_R, sci_list_I], mflat_list):
    preproc(sci_list, mbias.data, mdark_sci.data, mflat, rdnoise, show=True)
    
###############################################################################

# fetching path list of the preprocessed science frames
sci_list_B  = list(DATADIR.glob('p' + OBJNAME + '*B.fits'))
sci_list_I  = list(DATADIR.glob('p' + OBJNAME + '*I.fits'))
sci_list_R  = list(DATADIR.glob('p' + OBJNAME + '*R.fits'))
sci_list_V  = list(DATADIR.glob('p' + OBJNAME + '*V.fits'))

sci_list = sci_list_R # here I just selected only the I-band images.

# for displaying in zscale
interval = ZScaleInterval()

fig, axs = plt.subplots(1,len(sci_list), figsize=(3*len(sci_list),3))

for i in range(int(len(sci_list))):
    img = fits.getdata(sci_list[i], ext=0)
    trim = img[760:860, 1063:1163]
    vmin, vmax = interval.get_limits(trim)
    ax = axs[i]
    ax.imshow(trim, origin='lower', vmin=vmin, vmax=vmax)
    if i == 0:
        # find the rough peak coordinate
        xc, yc = np.unravel_index(np.argmax(trim), trim.shape)
    ax.axvline(yc, c='r', ls='--')
    ax.axhline(xc, c='r', ls='--')
    
###############################################################################

# reference image: frame 0
id_ref = 0
dat_ref, hdr_ref = fits.getdata(sci_list[id_ref], header=True, ext=0)
ccd_ref = CCDData(dat_ref, unit='adu')

# Aligning other images with respect to the reference image
start_time = time.time()
aligned_list = []
for i in range(len(sci_list)):
    dat = fits.getdata(sci_list[i], ext=0)
    msk = fits.getdata(sci_list[i], ext=1)
    ccd = CCDData(dat.byteswap().newbyteorder(), unit='adu', mask=msk)
    if (i == id_ref):
        ccd_aligned = ccd
    else:
        dat_aligned, footprint = aa.register(ccd, ccd_ref,
                                             max_control_points=50,
                                             detection_sigma=5, min_area=16,
                                             propagate_mask=True)
        ccd_aligned = CCDData(dat_aligned, unit='adu')
    aligned_list.append(ccd_aligned)
    # fits.writeto(imglist[i].split(".fit")[0]+"_align_auto.fits", dat_aligned, overwrite=True)
end_time = time.time()
print(f"--- {end_time-start_time:.4f} sec were taken for aligning {len(sci_list):d} images ---") 

fig, axs = plt.subplots(1,len(sci_list), figsize=(3*len(sci_list),3))
for i in range(len(aligned_list)):
    img = aligned_list[i].data
    trim = img[760:860, 1063:1163]
    vmin, vmax = interval.get_limits(trim)
    ax = axs[i]
    ax.imshow(trim, origin='lower', vmin=vmin, vmax=vmax)
    if i == 0:
        xc, yc = np.unravel_index(np.argmax(trim), trim.shape)
    ax.axvline(yc, c='r', ls='--')
    ax.axhline(xc, c='r', ls='--')
    
combined = combine(aligned_list, sigma_clip=True,
                   sigma_clip_high_thresh=3, sigma_clip_low_thresh=3)

plt.figure(figsize=(9,6))
vmin, vmax = interval.get_limits(combined.data)
plt.imshow(combined, origin='lower', vmax=vmax, vmin=vmin)

band = hdr_ref['FILTER']
hdr_ref['CFRAMES'] = len(sci_list)
hdr_ref['history'] = f'Combined image of {len(sci_list)} frames'
now = time.strftime("%Y-%m-%d %H:%M:%S") ##(%Z = GMT%z)
hdr_ref['history'] = f'Combined at {now}'
scihdu = fits.PrimaryHDU(data=combined.data.astype('float32'), header=hdr_ref)
mskhdu = fits.ImageHDU(data=combined.mask.astype('int'), name='MASK')
hdul = fits.HDUList([scihdu, mskhdu])
hdul.writeto(DATADIR/(OBJNAME+'_pre_'+band+'.fits'), overwrite=True)    