import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import binary_dilation
import os
import SimpleITK as sitk
import cv2 as cv
from scipy import ndimage
# from vesselness2d import vesselness2d?

def savenii(image,refer,outfile,dtype = 'Float32'):
    savefile = sitk.GetImageFromArray(image)
    savefile.SetSpacing(refer.GetSpacing())
    savefile.SetDirection(refer.GetDirection())
    savefile.SetOrigin(refer.GetOrigin())
    if dtype == 'Float32':
        savefile = sitk.Cast(savefile, sitk.sitkFloat32)
    elif dtype == 'UInt8':
        savefile = sitk.Cast(savefile, sitk.sitkUInt8)
    else:
        print('please check for dtype of correct this funtion')
    sitk.WriteImage(savefile, outfile)

def N4(targetpath,outpath):
    if os.path.exists(targetpath):
        input_image = sitk.ReadImage(targetpath)
        corrected_images = []
        if len(input_image.GetSize()) == 4:
            for i in range(input_image.GetSize()[3]):
                image = input_image[:, :, :, i]
#                 corrected_image = sitk.N4BiasFieldCorrection(image)
                mask_image = sitk.OtsuThreshold(image,0,1,200)
                image = sitk.Cast(image, sitk.sitkFloat32)
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrected_image = corrector.Execute(image,mask_image)
                corrected_images.append(corrected_image)

    #         asls_N4 = sitk.ComposeImageFilter().Execute(corrected_images)
            asls_N4 = sitk.JoinSeries(corrected_images)
            sitk.WriteImage(asls_N4, outpath)
        elif len(input_image.GetSize()) == 3:
            mask_image = sitk.OtsuThreshold(input_image,0,1,200)
            input_image = sitk.Cast(input_image, sitk.sitkFloat32)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            output_image = corrector.Execute(input_image,mask_image)
            output_image = sitk.Cast(output_image, sitk.sitkFloat32)
            sitk.WriteImage(output_image, outpath)
        else: 
            print("please check for the dimension of input image!")
    else:
        print("No file: {}".format(targetpath))


def Plot3D(cbf, name=None, vmin = None, vmax=None, color=None, layout = 2):


    if vmax is None:
        vmax = 500
    if vmin is None:
        vmin = 0
    if color is None:
        color = 'gray'
    

    if cbf.ndim != 3:
        print("输入维度应为3D")
        return
    

    slices = np.linspace(cbf.shape[layout]*1/10, cbf.shape[layout]*9/10, 8, dtype=int)
    

    fig = plt.figure(figsize=(18,4))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 8), axes_pad=0.1, cbar_location="right", cbar_mode="single")
    
    for i, slice_idx in enumerate(slices):
        ax = grid[i]
        if layout == 0:
            im = ax.imshow(cbf[slice_idx,:,:], vmin=vmin, vmax=vmax, cmap=color)
        elif layout == 1:
            im = ax.imshow(cbf[:,slice_idx,:], vmin=vmin, vmax=vmax, cmap=color)
        elif layout == 2:
            im = ax.imshow(cbf[:,:,slice_idx], vmin=vmin, vmax=vmax, cmap=color)
        ax.set_axis_off()
        ax.set_title(f"{name} - Slice {slice_idx}",fontsize = '10')
    cbar = ax.cax.colorbar(im,ticks=np.linspace(0, vmax, 5))

    plt.show()