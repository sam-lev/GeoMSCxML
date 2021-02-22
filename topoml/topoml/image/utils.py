# Standard library imports

# Third party imports
import os
import gzip
import shutil
import tarfile
import imageio
import zipfile
import numpy as np
from skimage import morphology, exposure
from skimage.io import imsave
from skimage.measure import block_reduce
from skimage import morphology

from skimage import data
from numpy.lib import NumpyVersion
import scipy
from scipy import ndimage as ndi
from skimage.transform._geometric import (SimilarityTransform, AffineTransform,
                         ProjectiveTransform)
from skimage.transform._warps_cy import _warp_fast

from skimage._shared.utils import (get_bound_method_class, safe_as_int, warn,
                             convert_to_float)#, _to_ndimage_mode, _validate_interpolation_order)

from PIL import Image
# Local application imports
from topoml.image.feature import gaussian_blur_filter


def make_arc_image(image, msc,labeled_accuracy=1, invert=False):
    arc_mask_image = np.zeros(image.shape)
    print("SHAPEEEEE")
    print(image.shape)
    x = 0 if not invert else 1
    y = 1 if not invert else 0
    for a in msc.arcs:
        points = np.array(a.line)
        for point in points:
            arc_mask_image[int(point[x]), int(point[y])] = a.label_accuracy if a.label_accuracy is not None else 1
    return arc_mask_image


def make_mc_arc_image(image, msc, X=None, Y=None, invert=False):
    if X is None and Y is None:
        arc_mask_image = np.zeros(image.shape)
    else:
        arc_mask_image = np.zeros((X,Y))
    mask_index = 2 if invert else 0

    x = 0 if invert else 1
    y = 1 if invert else 0

    for a in msc.arcs:
        if mask_index not in [
            msc.nodes[a.node_ids[0]].index,
            msc.nodes[a.node_ids[1]].index,
        ]:
            points = np.array(a.line)
            for point in points:
                arc_mask_image[int(point[x]), int(point[y])] = a.label_accuracy if a.label_accuracy is not None else 1
    return arc_mask_image


def make_dilated_arc_image(image, msc, width,labeled_accuracy=1, invert=True):
    return morphology.dilation(
        make_arc_image(image, msc,labeled_accuracy=labeled_accuracy, invert=invert), selem=morphology.disk(width)
    )


def make_arc_mask(image, msc, labeled_accuracy=1, invert=False):
    arc_mask_image = make_arc_image(image, msc, labeled_accuracy=labeled_accuracy,invert=invert)
    return np.ma.masked_where(arc_mask_image == 0, arc_mask_image)


def make_mc_arc_mask(image, msc, X=None, Y=None, labeled_accuracy=1, invert=False):
    if image is not None:
        X=image.shape[0]
        Y=image.shape[1]
    arc_mask_image = make_mc_arc_image(image=image, msc=msc, X=X, Y=Y, invert=invert)
    return np.ma.masked_where(arc_mask_image == 0, arc_mask_image)


def blur_and_save(original_image, fname_base, blur_sigma=2, grey_scale=True):
    blurred_image = gaussian_blur_filter(original_image, sigma=blur_sigma, as_grey=grey_scale).astype(
        "float32"
    )
    fname_raw = fname_base + "_smoothed.raw"
    blurred_image.tofile(fname_raw)
    return blurred_image, fname_raw

def augment_channels(original_image, fname_base, channels = [0,1]):
    import copy
    import cv2
    augmented_image = copy.deepcopy(original_image)
    
    #[ 0 = blue, 1 = green, 2 = red ]
    for c in channels:
        augmented_image[:,:,c] = 0#cv2.equalizeHist(augmented_image[:,:,c])
    fname_components = fname_base.split('.')
    fname_aug = fname_components[0]+"_aug."+fname_components[1]
    imsave(fname_aug, augmented_image)
    return augmented_image, fname_aug

def scale_intensity(original_image, fname_base, scale_range=(0,255)):
    scaled_image = exposure.rescale_intensity(original_image)#, in_range=(0, 255))
    fname_raw = fname_base + "_scaled.raw"
    scaled_image.tofile(fname_raw)
    return scaled_image, fname_raw  


def bounding_box(points):
    points = np.array(points)
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])

    return (xmin, xmax, ymin, ymax)


def range_overlap(a_min, a_max, b_min, b_max):
    return (a_min <= b_max) and (b_min <= a_max)


def box_intersection(b1, b2):
    return range_overlap(b1[0], b1[1], b2[0], b2[1]) and range_overlap(
        b1[2], b1[3], b2[2], b2[3]
    )

#resize the images of the dataset to be half the height and half the width of the original images, so
# that models states can fit on the GPU memory
def resize_img(img):
    if len(img.shape)==3:
        img = np.array(Image.fromarray(img).resize(((img.shape[1]+1)//2,(img.shape[0]+1)//2), Image.BILINEAR))
    else:
        img = block_reduce(img, block_size=(2, 2), func=np.max)
    return img

#delete small regions (<size) of binary images
def remove_small_regions(img, size):
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img



### for image to polar  ############################################################
def _fix_ndimage_mode(mode):
    # SciPy 1.6.0 introduced grid variants of constant and wrap which
    # have less surprising behavior for images. Use these when available
    grid_modes = {'constant': 'grid-constant', 'wrap': 'grid-wrap'}
    if NumpyVersion(scipy.__version__) >= '1.6.0':
        mode = grid_modes.get(mode, mode)
    return mode

def _to_ndimage_mode(mode):
    """Convert from `numpy.pad` mode name to the corresponding ndimage mode."""
    mode_translation_dict = dict(constant='constant', edge='nearest',
                                 symmetric='reflect', reflect='mirror',
                                 wrap='wrap')
    if mode not in mode_translation_dict:
        raise ValueError(
            ("Unknown mode: '{}', or cannot translate mode. The "
             "mode should be one of 'constant', 'edge', 'symmetric', "
             "'reflect', or 'wrap'. See the documentation of numpy.pad for"
             "more info.").format(mode))
    return _fix_ndimage_mode(mode_translation_dict[mode])

def _validate_interpolation_order(image_dtype, order):
    """Validate and return spline interpolation's order.
    Parameters
    ----------
    image_dtype : dtype
        Image dtype.
    order : int, optional
        The order of the spline interpolation. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    Returns
    -------
    order : int
        if input order is None, returns 0 if image_dtype is bool and 1
        otherwise. Otherwise, image_dtype is checked and input order
        is validated accordingly (order > 0 is not supported for bool
        image dtype)
    """

    if order is None:
        return 0 if image_dtype == bool else 1

    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the "
                         "range 0-5.")

    if image_dtype == bool and order != 0:
        warn("Input image dtype is bool. Interpolation is not defined "
             "with bool data type. Please set order to 0 or explicitely "
             "cast input image to another data type. Starting from version "
             "0.19 a ValueError will be raised instead of this warning.",
             FutureWarning, stacklevel=2)

    return order

HOMOGRAPHY_TRANSFORMS = (
    SimilarityTransform,
    AffineTransform,
    ProjectiveTransform
)


def resize(image, output_shape, order=None, mode='reflect', cval=0, clip=True,
           preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None):
    """Resize image to match a certain size.
    Performs interpolation to up-size or down-size N-dimensional images. Note
    that anti-aliasing should be enabled when down-sizing images to avoid
    aliasing artifacts. For down-sampling with an integer factor also see
    `skimage.transform.downscale_local_mean`.
    Parameters
    ----------
    image : ndarray
        Input image.
    output_shape : tuple or ndarray
        Size of the generated output image `(rows, cols[, ...][, dim])`. If
        `dim` is not provided, the number of channels is preserved. In case the
        number of input channels does not equal the number of output channels a
        n-dimensional interpolation is applied.
    Returns
    -------
    resized : ndarray
        Resized version of the input.
    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior
        to down-scaling. It is crucial to filter when down-sampling
        the image to avoid aliasing artifacts. If input image data
        type is bool, no anti-aliasing is applied.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering to avoid aliasing artifacts.
        By default, this value is chosen as (s - 1) / 2 where s is the
        down-scaling factor, where s > 1. For the up-size case, s < 1, no
        anti-aliasing is performed prior to rescaling.
    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].
    (100, 100)
    """
    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        # append dimensions to input_shape
        input_shape = input_shape + (1, ) * (output_ndim - image.ndim)
        image = np.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        # multichannel case: append shape of last axis
        output_shape = output_shape + (image.shape[-1], )
    elif output_ndim < image.ndim - 1:
        raise ValueError("len(output_shape) cannot be smaller than the image "
                         "dimensions")

    if anti_aliasing is None:
        anti_aliasing = not image.dtype == bool

    if image.dtype == bool and anti_aliasing:
        warn("Input image dtype is bool. Gaussian convolution is not defined "
             "with bool data type. Please set anti_aliasing to False or "
             "explicitely cast input image to another data type. Starting "
             "from version 0.19 a ValueError will be raised instead of this "
             "warning.", FutureWarning, stacklevel=2)

    factors = (np.asarray(input_shape, dtype=float) /
               np.asarray(output_shape, dtype=float))

    # Translate modes used by np.pad to those used by scipy.ndimage
    ndi_mode = _to_ndimage_mode(mode)
    if anti_aliasing:
        if anti_aliasing_sigma is None:
            anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
        else:
            anti_aliasing_sigma = \
                np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
            if np.any(anti_aliasing_sigma < 0):
                raise ValueError("Anti-aliasing standard deviation must be "
                                 "greater than or equal to zero")
            elif np.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                warn("Anti-aliasing standard deviation greater than zero but "
                     "not down-sampling along all axes")
        image = ndi.gaussian_filter(image, anti_aliasing_sigma,
                                    cval=cval, mode=ndi_mode)

    if NumpyVersion(scipy.__version__) >= '1.6.0':
        # The grid_mode kwarg was introduced in SciPy 1.6.0
        order = _validate_interpolation_order(image.dtype, order)
        zoom_factors = [1 / f for f in factors]
        image = convert_to_float(image, preserve_range)
        out = ndi.zoom(image, zoom_factors, order=order, mode=ndi_mode,
                       cval=cval, grid_mode=True)
        _clip_warp_output(image, out, order, mode, cval, clip)

    # TODO: Remove the fallback code below once SciPy >= 1.6.0 is required.

    # 2-dimensional interpolation
    elif len(output_shape) == 2 or (len(output_shape) == 3 and
                                  output_shape[2] == input_shape[2]):
        rows = output_shape[0]
        cols = output_shape[1]
        input_rows = input_shape[0]
        input_cols = input_shape[1]
        if rows == 1 and cols == 1:
            tform = AffineTransform(translation=(input_cols / 2.0 - 0.5,
                                                 input_rows / 2.0 - 0.5))
        else:
            # 3 control points necessary to estimate exact AffineTransform
            src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
            dst_corners = np.zeros(src_corners.shape, dtype=np.double)
            # take into account that 0th pixel is at position (0.5, 0.5)
            dst_corners[:, 0] = factors[1] * (src_corners[:, 0] + 0.5) - 0.5
            dst_corners[:, 1] = factors[0] * (src_corners[:, 1] + 0.5) - 0.5

            tform = AffineTransform()
            tform.estimate(src_corners, dst_corners)

        # Make sure the transform is exactly metric, to ensure fast warping.
        tform.params[2] = (0, 0, 1)
        tform.params[0, 1] = 0
        tform.params[1, 0] = 0

        out = warp(image, tform, output_shape=output_shape, order=order,
                   mode=mode, cval=cval, clip=clip,
                   preserve_range=preserve_range)

    else:  # n-dimensional interpolation
        order = _validate_interpolation_order(image.dtype, order)

        coord_arrays = [factors[i] * (np.arange(d) + 0.5) - 0.5
                        for i, d in enumerate(output_shape)]

        coord_map = np.array(np.meshgrid(*coord_arrays,
                                         sparse=False,
                                         indexing='ij'))

        image = convert_to_float(image, preserve_range)

        out = ndi.map_coordinates(image, coord_map, order=order,
                                  mode=ndi_mode, cval=cval)

        _clip_warp_output(image, out, order, mode, cval, clip)

    return out


def rescale(image, scale, order=None, mode='reflect', cval=0, clip=True,
            preserve_range=False, multichannel=False,
            anti_aliasing=None, anti_aliasing_sigma=None):
    """Scale image by a certain factor.
    Performs interpolation to up-scale or down-scale N-dimensional images.
    Note that anti-aliasing should be enabled when down-sizing images to avoid
    aliasing artifacts. For down-sampling with an integer factor also see
    `skimage.transform.downscale_local_mean`.
    Parameters
    ----------
    image : ndarray
        Input image.
    scale : {float, tuple of floats}
        Scale factors. Separate scale factors can be defined as
        `(rows, cols[, ...][, dim])`.
    Returns
    -------
    scaled : ndarray
        Scaled version of the input.
    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior
        to down-scaling. It is crucial to filter when down-sampling
        the image to avoid aliasing artifacts. If input image data
        type is bool, no anti-aliasing is applied.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering to avoid aliasing artifacts.
        By default, this value is chosen as (s - 1) / 2 where s is the
        down-scaling factor.
    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].
    Examples
    (256, 256)
    """
    scale = np.atleast_1d(scale)
    if len(scale) > 1:
        if ((not multichannel and len(scale) != image.ndim) or
                (multichannel and len(scale) != image.ndim - 1)):
            raise ValueError("Supply a single scale, or one value per spatial "
                             "axis")
        if multichannel:
            scale = np.concatenate((scale, [1]))
    orig_shape = np.asarray(image.shape)
    output_shape = np.round(scale * orig_shape)
    if multichannel:  # don't scale channel dimension
        output_shape[-1] = orig_shape[-1]

    return resize(image, output_shape, order=order, mode=mode, cval=cval,
                  clip=clip, preserve_range=preserve_range,
                  anti_aliasing=anti_aliasing,
                  anti_aliasing_sigma=anti_aliasing_sigma)


def rotate(image, angle, resize=False, center=None, order=None,
           mode='constant', cval=0, clip=True, preserve_range=False):
    """Rotate image by a certain angle around its center.
    Parameters
    ----------
    image : ndarray
        Input image.
    angle : float
        Rotation angle in degrees in counter-clockwise direction.
    resize : bool, optional
        Determine whether the shape of the output image will be automatically
        calculated, so the complete rotated image exactly fits. Default is
        False.
    center : iterable of length 2
        The rotation center. If ``center=None``, the image is rotated around
        its center, i.e. ``center=(cols / 2 - 0.5, rows / 2 - 0.5)``.  Please
        note that this parameter is (cols, rows), contrary to normal skimage
        ordering.
    Returns
    -------
    rotated : ndarray
        Rotated version of the input.
    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].
    Examples
    --------
    (512, 512)
    """

    rows, cols = image.shape[0], image.shape[1]

    # rotation around center
    if center is None:
        center = np.array((cols, rows)) / 2. - 0.5
    else:
        center = np.asarray(center)
    tform1 = SimilarityTransform(translation=center)
    tform2 = SimilarityTransform(rotation=np.deg2rad(angle))
    tform3 = SimilarityTransform(translation=-center)
    tform = tform3 + tform2 + tform1

    output_shape = None
    if resize:
        # determine shape of output image
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.around((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = SimilarityTransform(translation=translation)
        tform = tform4 + tform

    # Make sure the transform is exactly affine, to ensure fast warping.
    tform.params[2] = (0, 0, 1)

    return warp(image, tform, output_shape=output_shape, order=order,
                mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)


def downscale_local_mean(image, factors, cval=0, clip=True):
    """Down-sample N-dimensional image by local averaging.
    The image is padded with `cval` if it is not perfectly divisible by the
    integer factors.
    In contrast to interpolation in `skimage.transform.resize` and
    `skimage.transform.rescale` this function calculates the local mean of
    elements in each block of size `factors` in the input image.
    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    factors : array_like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        integer factors.
    clip : bool, optional
        Unused, but kept here for API consistency with the other transforms
        in this module. (The local mean will never fall outside the range
        of values in the input image, assuming the provided `cval` also
        falls within that range.)
    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.
        For integer inputs, the output dtype will be ``float64``.
        See :func:`numpy.mean` for details.
    Examples
    """
    return block_reduce(image, factors, np.mean, cval)


def _swirl_mapping(xy, center, rotation, strength, radius):
    x, y = xy.T
    x0, y0 = center
    rho = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Ensure that the transformation decays to approximately 1/1000-th
    # within the specified radius.
    radius = radius / 5 * np.log(2)

    theta = rotation + strength * \
        np.exp(-rho / radius) + \
        np.arctan2(y - y0, x - x0)

    xy[..., 0] = x0 + rho * np.cos(theta)
    xy[..., 1] = y0 + rho * np.sin(theta)

    return xy


def swirl(image, center=None, strength=1, radius=100, rotation=0,
          output_shape=None, order=None, mode='reflect', cval=0, clip=True,
          preserve_range=False):
    """Perform a swirl transformation.
    Parameters
    ----------
    image : ndarray
        Input image.
    center : (column, row) tuple or (2,) ndarray, optional
        Center coordinate of transformation.
    strength : float, optional
        The amount of swirling applied.
    radius : float, optional
        The extent of the swirl in pixels.  The effect dies out
        rapidly beyond `radius`.
    rotation : float, optional
        Additional rotation applied to the image.
    Returns
    -------
    swirled : ndarray
        Swirled version of the input.
    Other parameters
    ----------------
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved.
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode, with 'constant' used as the default. Modes match
        the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    """
    if center is None:
        center = np.array(image.shape)[:2][::-1] / 2

    warp_args = {'center': center,
                 'rotation': rotation,
                 'strength': strength,
                 'radius': radius}

    return warp(image, _swirl_mapping, map_args=warp_args,
                output_shape=output_shape, order=order, mode=mode, cval=cval,
                clip=clip, preserve_range=preserve_range)


def _stackcopy(a, b):
    """Copy b into each color layer of a, such that::
      a[:,:,0] = a[:,:,1] = ... = b
    Parameters
    ----------
    a : (M, N) or (M, N, P) ndarray
        Target array.
    b : (M, N)
        Source array.
    Notes
    -----
    Color images are stored as an ``(M, N, 3)`` or ``(M, N, 4)`` arrays.
    """
    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b


def warp_coords(coord_map, shape, dtype=np.float64):
    """Build the source coordinates for the output of a 2-D image warp.
    Parameters
    ----------
    coord_map : callable like GeometricTransform.inverse
        Return input coordinates for given output coordinates.
        Coordinates are in the shape (P, 2), where P is the number
        of coordinates and each element is a ``(row, col)`` pair.
    shape : tuple
        Shape of output image ``(rows, cols[, bands])``.
    dtype : np.dtype or string
        dtype for return value (sane choices: float32 or float64).
    Returns
    -------
    coords : (ndim, rows, cols[, bands]) array of dtype `dtype`
            Coordinates for `scipy.ndimage.map_coordinates`, that will yield
            an image of shape (orows, ocols, bands) by drawing from source
            points according to the `coord_transform_fn`.
    Notes
    -----
    This is a lower-level routine that produces the source coordinates for 2-D
    images used by `warp()`.
    It is provided separately from `warp` to give additional flexibility to
    users who would like, for example, to re-use a particular coordinate
    mapping, to use specific dtypes at various points along the the
    image-warping process, or to implement different post-processing logic
    than `warp` performs after the call to `ndi.map_coordinates`.
    Examples
    --------
    Produce a coordinate map that shifts an image up and to the right:
    """
    shape = safe_as_int(shape)
    rows, cols = shape[0], shape[1]
    coords_shape = [len(shape), rows, cols]
    if len(shape) == 3:
        coords_shape.append(shape[2])
    coords = np.empty(coords_shape, dtype=dtype)

    # Reshape grid coordinates into a (P, 2) array of (row, col) pairs
    tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T

    # Map each (row, col) pair to the source image according to
    # the user-provided mapping
    tf_coords = coord_map(tf_coords)

    # Reshape back to a (2, M, N) coordinate grid
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    # Place the y-coordinate mapping
    _stackcopy(coords[1, ...], tf_coords[0, ...])

    # Place the x-coordinate mapping
    _stackcopy(coords[0, ...], tf_coords[1, ...])

    if len(shape) == 3:
        coords[2, ...] = range(shape[2])

    return coords


def _clip_warp_output(input_image, output_image, order, mode, cval, clip):
    """Clip output image to range of values of input image.
    Note that this function modifies the values of `output_image` in-place
    and it is only modified if ``clip=True``.
    Parameters
    ----------
    input_image : ndarray
        Input image.
    output_image : ndarray
        Output image, which is modified in-place.
    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    """
    if clip and order != 0:
        min_val = input_image.min()
        max_val = input_image.max()

        preserve_cval = (mode == 'constant' and not
                         (min_val <= cval <= max_val))

        if preserve_cval:
            cval_mask = output_image == cval

        np.clip(output_image, min_val, max_val, out=output_image)

        if preserve_cval:
            output_image[cval_mask] = cval


def warp(image, inverse_map, map_args={}, output_shape=None, order=None,
         mode='constant', cval=0., clip=True, preserve_range=False):
    """Warp an image according to a given coordinate transformation.
    Parameters
    ----------
    image : ndarray
        Input image.
    inverse_map : transformation object, callable ``cr = f(cr, **kwargs)``, or ndarray
        Inverse coordinate map, which transforms coordinates in the output
        images into their corresponding coordinates in the input image.
        There are a number of different options to define this map, depending
        on the dimensionality of the input image. A 2-D image can have 2
        dimensions for gray-scale images, or 3 dimensions with color
        information.
         - For 2-D images, you can directly pass a transformation object,
           e.g. `skimage.transform.SimilarityTransform`, or its inverse.
         - For 2-D images, you can pass a ``(3, 3)`` homogeneous
           transformation matrix, e.g.
           `skimage.transform.SimilarityTransform.params`.
         - For 2-D images, a function that transforms a ``(M, 2)`` array of
           ``(col, row)`` coordinates in the output image to their
           corresponding coordinates in the input image. Extra parameters to
           the function can be specified through `map_args`.
         - For N-D images, you can directly pass an array of coordinates.
           The first dimension specifies the coordinates in the input image,
           while the subsequent dimensions determine the position in the
           output image. E.g. in case of 2-D images, you need to pass an array
           of shape ``(2, rows, cols)``, where `rows` and `cols` determine the
           shape of the output image, and the first dimension contains the
           ``(row, col)`` coordinate in the input image.
           See `scipy.ndimage.map_coordinates` for further documentation.
        Note, that a ``(3, 3)`` matrix is interpreted as a homogeneous
        transformation matrix, so you cannot interpolate values from a 3-D
        input, if the output is of shape ``(3,)``.
        See example section for usage.
    map_args : dict, optional
        Keyword arguments passed to `inverse_map`.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved.  Note that, even for multi-band images, only rows
        and columns need to be specified.
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5:
         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic
         Default is 0 if image.dtype is bool and 1 otherwise.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    Returns
    -------
    warped : double ndarray
        The warped input image.
    Notes
    -----
    - The input image is converted to a `double` image.
    - In case of a `SimilarityTransform`, `AffineTransform` and
      `ProjectiveTransform` and `order` in [0, 3] this function uses the
      underlying transformation matrix to warp the image with a much faster
      routine.
    Examples
    --------

    """

    if image.size == 0:
        raise ValueError("Cannot warp empty image with dimensions",
                         image.shape)

    order = _validate_interpolation_order(image.dtype, order)

    image = convert_to_float(image, preserve_range)

    input_shape = np.array(image.shape)

    if output_shape is None:
        output_shape = input_shape
    else:
        output_shape = safe_as_int(output_shape)

    warped = None

    if order == 2:
        # When fixing this issue, make sure to fix the branches further
        # below in this function
        warn("Bi-quadratic interpolation behavior has changed due "
             "to a bug in the implementation of scikit-image. "
             "The new version now serves as a wrapper "
             "around SciPy's interpolation functions, which itself "
             "is not verified to be a correct implementation. Until "
             "skimage's implementation is fixed, we recommend "
             "to use bi-linear or bi-cubic interpolation instead.")

    if order in (0, 1, 3) and not map_args:
        # use fast Cython version for specific interpolation orders and input

        matrix = None

        if isinstance(inverse_map, np.ndarray) and inverse_map.shape == (3, 3):
            # inverse_map is a transformation matrix as numpy array
            matrix = inverse_map

        elif isinstance(inverse_map, HOMOGRAPHY_TRANSFORMS):
            # inverse_map is a homography
            matrix = inverse_map.params

        elif (hasattr(inverse_map, '__name__') and
              inverse_map.__name__ == 'inverse' and
              get_bound_method_class(inverse_map) in HOMOGRAPHY_TRANSFORMS):
            # inverse_map is the inverse of a homography
            matrix = np.linalg.inv(inverse_map.__self__.params)

        if matrix is not None:
            matrix = matrix.astype(image.dtype)
            ctype = 'float32_t' if image.dtype == np.float32 else 'float64_t'
            if image.ndim == 2:
                warped = _warp_fast[ctype](image, matrix,
                                           output_shape=output_shape,
                                           order=order, mode=mode, cval=cval)
            elif image.ndim == 3:
                dims = []
                for dim in range(image.shape[2]):
                    dims.append(_warp_fast[ctype](image[..., dim], matrix,
                                                  output_shape=output_shape,
                                                  order=order, mode=mode,
                                                  cval=cval))
                warped = np.dstack(dims)

    if warped is None:
        # use ndi.map_coordinates

        if (isinstance(inverse_map, np.ndarray) and
                inverse_map.shape == (3, 3)):
            # inverse_map is a transformation matrix as numpy array,
            # this is only used for order >= 4.
            inverse_map = ProjectiveTransform(matrix=inverse_map)

        if isinstance(inverse_map, np.ndarray):
            # inverse_map is directly given as coordinates
            coords = inverse_map
        else:
            # inverse_map is given as function, that transforms (N, 2)
            # destination coordinates to their corresponding source
            # coordinates. This is only supported for 2(+1)-D images.

            if image.ndim < 2 or image.ndim > 3:
                raise ValueError("Only 2-D images (grayscale or color) are "
                                 "supported, when providing a callable "
                                 "`inverse_map`.")

            def coord_map(*args):
                return inverse_map(*args, **map_args)

            if len(input_shape) == 3 and len(output_shape) == 2:
                # Input image is 2D and has color channel, but output_shape is
                # given for 2-D images. Automatically add the color channel
                # dimensionality.
                output_shape = (output_shape[0], output_shape[1],
                                input_shape[2])

            coords = warp_coords(coord_map, output_shape)

        # Pre-filtering not necessary for order 0, 1 interpolation
        prefilter = order > 1

        ndi_mode = _to_ndimage_mode(mode)
        warped = ndi.map_coordinates(image, coords, prefilter=prefilter,
                                     mode=ndi_mode, order=order, cval=cval)

    _clip_warp_output(image, warped, order, mode, cval, clip)

    return warped


def _linear_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesian to polar coordinates
    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = ncols / radius``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.
    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((output_coords[:, 0] / k_radius) * np.sin(angle)) + center[0]
    cc = ((output_coords[:, 0] / k_radius) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords


def _log_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesian to polar coordinates
    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = width / np.log(radius)``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.
    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((np.exp(output_coords[:, 0] / k_radius)) * np.sin(angle)) + center[0]
    cc = ((np.exp(output_coords[:, 0] / k_radius)) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords


def warp_polar(image, center=None, *, radius=None, output_shape=None,
               scaling='linear', multichannel=False, **kwargs):
    """Remap image to polar or log-polar coordinates space.
    Parameters
    ----------
    image : ndarray
        Input image. Only 2-D arrays are accepted by default. If
        `multichannel=True`, 3-D arrays are accepted and the last axis is
        interpreted as multiple channels.
    center : tuple (row, col), optional
        Point in image that represents the center of the transformation (i.e.,
        the origin in cartesian space). Values can be of type `float`.
        If no value is given, the center is assumed to be the center point
        of the image.
    radius : float, optional
        Radius of the circle that bounds the area to be transformed.
    output_shape : tuple (row, col), optional
    scaling : {'linear', 'log'}, optional
        Specify whether the image warp is polar or log-polar. Defaults to
        'linear'.
    multichannel : bool, optional
        Whether the image is a 3-D array in which the third axis is to be
        interpreted as multiple channels. If set to `False` (default), only 2-D
        arrays are accepted.
    **kwargs : keyword arguments
        Passed to `transform.warp`.
    Returns
    -------
    warped : ndarray
        The polar or log-polar warped image.
    Examples
    --------
    """
    if image.ndim != 2 and not multichannel:
        raise ValueError("Input array must be 2 dimensions "
                         "when `multichannel=False`,"
                         " got {}".format(image.ndim))

    if image.ndim != 3 and multichannel:
        raise ValueError("Input array must be 3 dimensions "
                         "when `multichannel=True`,"
                         " got {}".format(image.ndim))

    if center is None:
        center = (np.array(image.shape)[:2] / 2) - 0.5

    if radius is None:
        w, h = np.array(image.shape)[:2] / 2
        radius = np.sqrt(w ** 2 + h ** 2)

    if output_shape is None:
        height = 360
        width = int(np.ceil(radius))
        output_shape = (height, width)
    else:
        output_shape = safe_as_int(output_shape)
        height = output_shape[0]
        width = output_shape[1]

    if scaling == 'linear':
        k_radius = width / radius
        map_func = _linear_polar_mapping
    elif scaling == 'log':
        k_radius = width / np.log(radius)
        map_func = _log_polar_mapping
    else:
        raise ValueError("Scaling value must be in {'linear', 'log'}")

    k_angle = height / (2 * np.pi)
    warp_args = {'k_angle': k_angle, 'k_radius': k_radius, 'center': center}

    warped = warp(image, map_func, map_args=warp_args,
                  output_shape=output_shape, **kwargs)

    return warped