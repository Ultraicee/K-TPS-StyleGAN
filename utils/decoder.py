import tensorflow as tf
import numpy as np


def convert_images_to_255(images, drange=[-1, 1], nchw_to_nhwc=False, shrink=1):
    """
    Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    images = tf.cast(images, tf.float32)
    if shrink > 1:
        k_size = [1, 1, shrink, shrink]
        images = tf.nn.avg_pool(images, ksize=k_size, strides=k_size, padding="VALID", data_format="NCHW")
    if nchw_to_nhwc:
        images = tf.transpose(images, [0, 2, 3, 1])
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    return images


def convert_images_to_orinrange(images, drange=[-1, 1], nchw_to_nhwc=False, shrink=1):
    """
    Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    images = tf.cast(images, tf.float32)
    if shrink > 1:
        k_size = [1, 1, shrink, shrink]
        images = tf.nn.avg_pool(images, ksize=k_size, strides=k_size, padding="VALID", data_format="NCHW")
    if nchw_to_nhwc:
        images = tf.transpose(images, [0, 2, 3, 1])

    def adjust_dynamic_range(data, drange_in, drange_out):
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                        np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    images = adjust_dynamic_range(images, drange, [-50, 50])
    return images


def decoder_forward2(disp, sz_params):
    d_height, d_width = get_tps_size(sz_params)
    disp_tensor = tf.reshape(disp, [sz_params.data_size, d_height, d_width, 1])  # shape = [b h w 1]
    disp_ex = extend_cutted_disp(disp_tensor, sz_params)
    return disp_ex


def decoder_forward1(z_in, sz_params, disp, compensate_disp):
    disp_tensor = convert_images_to_orinrange(disp, nchw_to_nhwc=True)
    disp_tensor = disp_tensor + compensate_disp
    d_height, d_width = get_tps_size(sz_params)
    disp_tensor = tf.reshape(disp_tensor, [tf.shape(z_in)[0], d_height, d_width, 1])
    disp_ex = extend_cutted_disp(disp_tensor, sz_params)
    return disp_ex

def get_tps_size(sz_params):
    """
    sz_params = size_params(batch=50,
                            height=288,
                            width=360,
                            channel=3,
                            cutTop=20,
                            cutBottom=0,
                            cutLeft=0,
                            cutRight=50)
    """
    tps_height = np.int32(sz_params.height - sz_params.cutTop - sz_params.cutBottom)
    tps_width = np.int32(sz_params.width - sz_params.cutLeft - sz_params.cutRight)
    return tps_height, tps_width

def decoder_forward(z_in, T_matrix, sz_params):
    """
    Equal to a fully connected layer, the TPS matrix is multiplied by matrix multiplication 
    with the control points generated by the encoder to generate a parallax plot of the interpolated area
    The interpolation area is not the entire image area, the upper and right edges of the image 
    are cut off before interpolation
    :param z_in:
    :param T_matrix:
    :param sz_params:
    :return:
    """
    disp_vec = tf.map_fn(lambda x: tf.matmul(T_matrix, x), z_in)
    d_height, d_width = get_tps_size(sz_params)
    disp_tensor = tf.reshape(disp_vec, [tf.shape(z_in)[0], d_height, d_width, 1])
    disp_ex = extend_cutted_disp(disp_tensor, sz_params)
    return disp_ex

def extend_cutted_disp(disp, sz_params):
    """
    Fill the parallax plot of the interpolated area with zeros on the top and right edges
    to restore it to the size of the entire image. make it easier to reconstruct the right image
    with linear interpolation behind it
    :param disp:
    :param sz_params:
    :return:
    """
    d_height, d_width = get_tps_size(sz_params)
    ex_up = tf.zeros([tf.shape(disp)[0], sz_params.cutTop, d_width, tf.shape(disp)[3]])
    ex_bottom = tf.zeros([tf.shape(disp)[0], sz_params.cutBottom, d_width, tf.shape(disp)[3]])
    ex_left = tf.zeros(
        [tf.shape(disp)[0], sz_params.height, sz_params.cutLeft, tf.shape(disp)[3]])
    ex_right = tf.zeros(
        [tf.shape(disp)[0], sz_params.height, sz_params.cutRight, tf.shape(disp)[3]])

    ex_disp = tf.concat([ex_up, disp, ex_bottom], axis=1)
    ex_disp = tf.concat([ex_left, ex_disp, ex_right], axis=2)
    return ex_disp


def decoder_forward3(z_in, sz_params, disp, compensate_disp):
    disp_tensor = convert_images_to_orinrange(disp, nchw_to_nhwc=True)
    disp_tensor = disp_tensor + compensate_disp
    d_height, d_width = get_tps_size(sz_params)
    disp_tensor = tf.reshape(disp_tensor, [tf.shape(z_in)[0], d_height, d_width, 1])
    disp_ex = extend_cutted_disp(disp_tensor, sz_params)

    return disp_ex
