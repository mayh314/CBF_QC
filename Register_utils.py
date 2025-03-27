import ants

def Register(movefile, referfile, transform_dir, outputfile, metric='MI', transform='SyN', interpolator='bSpline'):
    """A registration function code using antspy

    Args:
        movefile (str): source image filename
        referfile (str): target image filename
        transform_dir (str): output transform matrix/field filename
        outputfile (str): warped source image filename
        metric (str, optional): similarity metric, including “MI”, “CC”, “demons”. Defaults to 'MI'.
        transform (str, optional): transform model, including “Rigid”, “Affine”, “SyN”. Defaults to 'SyN'.
        interpolator (str, optional): interpolator, including “bSpline”, “linear”, “nearestNeighbor”. Defaults to 'bSpline'.

    Return:
        None

    Raises:
        ValueError: Unavailable metric
        ValueError: Unavailable transform
    """

    moving = ants.image_read(movefile)
    fixed = ants.image_read(referfile)
    
    metrics = []
    if metric =='MI':
        metrics.append(['MI', fixed, moving, 1, 32])
    elif metric == 'CC':
        metrics.append(['CC', fixed, moving, 1, 5])
    elif metric == 'demons':
        metrics.append(['CC', fixed, moving, 1.5, 8])
        metrics.append(['demons', ants.iMath_grad(fixed), ants.iMath_grad(moving), 1, 1])
    else:
        raise ValueError("Unavailable metric!")
      
    if transform == 'Rigid' or transform == 'Affine':
        reg = ants.registration(fixed = fixed, 
                                moving = moving, 
                                type_of_transform = transform,  
                                multivariate_extras = metrics, 
                                outprefix=transform_dir
                                )
    elif transform == 'SyN':
        aff = ants.registration(fixed = fixed, 
                                moving = moving, 
                                type_of_transform = "Affine",  
                                multivariate_extras = metrics, 
                                outprefix = transform_dir)
        reg = ants.registration(fixed = fixed, 
                                moving = moving, 
                                type_of_transform = 'SyNOnly',  
                                multivariate_extras = metrics, 
                                initial_transform=aff['fwdtransforms'][0], 
                                outprefix=transform_dir
                                )
    else:
        raise ValueError("Unavailable transform!")
      
    output = ants.apply_transforms(fixed = fixed, 
                                   moving = moving, 
                                   transformlist = reg['fwdtransforms'],
                                   interpolator = interpolator
                                   )
    output.to_file(outputfile)
    return reg

def Register_apply_transform(movefile, referfile, transformlist, outputfile, interpolator='bSpline'):
    """A registration function code apply exist transform

    Args:
        movefile (str): source image filename
        referfile (str): target image filename
        transformlist (list): lists of transform matrix/field filename
        outputfile (str): warped source image filename
        interpolator (str, optional): interpolator, including “bSpline”, “linear”, “nearestNeighbor”. Defaults to 'bSpline'.
        
    Return:
        None
    """
    moving = ants.image_read(movefile)
    fixed = ants.image_read(referfile)
    output = ants.apply_transforms(fixed = fixed, 
                                   moving = moving, 
                                   transformlist = transformlist,
                                   interpolator = interpolator
                                   )
    output.to_file(outputfile)

# def Register_ANTS(movefile, referfile, transform_dir, outputfile):
#     cmd = 'ANTS 3 -m MI[{}, {}, 1, 32] -o {} -t Rigid -n BSpline'.format(referfile, movefile, transform_dir)
#     os.system(cmd)
#     cmd = 'WarpImageMultiTransform 3 {} {} -R {} {}Warp.nii {}Affine.txt --use-BSpline'.format(movefile, outputfile, referfile, transform_dir, transform_dir)
#     os.system(cmd)