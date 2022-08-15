import keras.backend as K
import tensorflow as tf
import numpy as np
from .decoder import get_tps_size


class LinearInterpolator:
    def __init__(self, sz_params):
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
        self.sz_params = sz_params
        self.batch = sz_params.mini_batch_size
        self.height = sz_params.height
        self.width = sz_params.width
        self.height_f = tf.cast(self.height, tf.float32)
        self.width_f = tf.cast(self.width, tf.float32)

        x_t, y_t = tf.meshgrid(tf.linspace(0.0, self.width_f - 1.0, self.width),
                               tf.linspace(0.0, self.height_f - 1.0, self.height))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))
        x_t_flat = tf.tile(x_t_flat, tf.stack([self.batch, 1]))
        y_t_flat = tf.tile(y_t_flat, tf.stack([self.batch, 1]))

        self.x_t_flat = tf.reshape(x_t_flat, [-1])
        self.y_t_flat = tf.reshape(y_t_flat, [-1])

    def _repeat(self, x, n_repeats):
        rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
        return tf.reshape(rep, [-1])

    def interpolate(self, img_source0, disp):
        """
        :param img_source: shape=[b, h_s, w_s, c],, dtype=tf.float32
        :param disp: shape=[b, h_s, w_s, 1]
        :return: shape=[b, h_s, w_s, c]
        """

        x = self.x_t_flat + tf.reshape(disp, [-1])  
        _edge_size = 1
        img_source = tf.pad(img_source0, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        x = x + _edge_size
        y = self.y_t_flat + _edge_size
        x = tf.clip_by_value(x, 0.0, self.width_f - 1 + 2 * _edge_size)  

        x0_f = tf.floor(x)
        y0_f = tf.floor(y)
        x1_f = x0_f + 1

        x0 = tf.cast(x0_f, tf.int32)
        y0 = tf.cast(y0_f, tf.int32)
        x1 = tf.cast(tf.minimum(x1_f, self.width_f - 1 + 2 * _edge_size), tf.int32)

        dim2 = (self.width + 2 * _edge_size)
        dim1 = (self.width + 2 * _edge_size) * (self.height + 2 * _edge_size)
        base = self._repeat(tf.range(self.batch) * dim1, self.height * self.width)
        base_y0 = base + y0 * dim2

        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_flat = tf.reshape(img_source, tf.stack([-1, tf.shape(img_source)[3]]))

        pix_l = tf.gather(im_flat, idx_l)
        pix_r = tf.gather(im_flat, idx_r)
        weight_l = tf.expand_dims(x1_f - x, 1)
        weight_r = tf.expand_dims(x - x0_f, 1)

        img_recon = weight_l * pix_l + weight_r * pix_r
        img_recon = tf.reshape(img_recon, tf.shape(img_source0))
        return img_recon
 
    
def compute_rec_loss_per(est_im, real_im, compensateI, left_im, sz_params, thres):
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
    d_height, d_width = get_tps_size(sz_params)
    est_clip = tf.slice(est_im, [0, sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                        name='est_r_clip')
    real_clip = tf.slice(real_im, [0, sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                         name='r_clip')
    left_clip = tf.slice(left_im, [0, sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                         name='l_clip')

    real_clip_gray = tf.image.rgb_to_grayscale(real_clip)
    left_clip_gray = tf.image.rgb_to_grayscale(left_clip)
    est_im_gray = tf.image.rgb_to_grayscale(est_clip)

    shape = tf.shape(real_clip_gray)
    reflect = tf.fill(shape, thres)  

    compa2 = tf.less(real_clip_gray, reflect)  
    compa3 = tf.less(est_im_gray, reflect)  

    compa4 = tf.greater(left_clip_gray, reflect)

    pad1 = np.array([[0, 0], [0, 0], [0, 100], [0, 0]])
    compa4_pad = tf.pad(compa4, pad1, name='pad_1')
    compa4_pad2 = tf.logical_not(compa4_pad, name='padcompa')

    compa3_1 = tf.slice(compa4_pad2, [0, 0, 67, 0], [-1, d_height, d_width, -1])  
    compa3_2 = tf.slice(compa4_pad2, [0, 0, 72, 0], [-1, d_height, d_width, -1])
    compa3_3 = tf.slice(compa4_pad2, [0, 0, 77, 0], [-1, d_height, d_width, -1])
    compa3_4 = tf.slice(compa4_pad2, [0, 0, 82, 0], [-1, d_height, d_width, -1])
    compa3_5 = tf.slice(compa4_pad2, [0, 0, 87, 0], [-1, d_height, d_width, -1])

    compa = tf.to_float(compa2 & compa3 & compa3_1 & compa3_2 & compa3_3 & compa3_4 & compa3_5)  

    kernel = np.array([
        [-1, -1, 0, 0, 0, -1, -1],
        [-1, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, -1],
        [-1, -1, 0, 0, 0, -1, -1]]).astype(np.float32).reshape(7, 7, 1)
    compa_erosion = tf.nn.erosion2d(compa, kernel, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                    padding="SAME")  
    compa_rgb = tf.tile(compa_erosion, [1, 1, 1, 3])

    loss_rec_sum = tf.reduce_sum(tf.multiply(tf.square(est_clip - real_clip - compensateI), compa_rgb), axis=[1, 2, 3])
    compa_sum = tf.reduce_sum(compa_rgb, axis=[1, 2, 3])
    loss_rec = tf.reduce_mean(tf.divide(loss_rec_sum, compa_sum))

    loss_rec_perimg = tf.reduce_mean(loss_rec_sum)
    compa_sum_batch = tf.reduce_sum(compa_sum)

    return loss_rec, compa_sum_batch, loss_rec_perimg


def compute_rec_loss(est_im, real_im, compensateI, left_im, sz_params, thres):
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
    d_height, d_width = get_tps_size(sz_params)
    est_clip = tf.slice(est_im, [0, sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                        name='est_r_clip')
    real_clip = tf.slice(real_im, [0, sz_params.cutTop, sz_params.cutLeft, 0], [-1, d_height, d_width, -1],
                         name='r_clip')

    real_clip_gray = tf.image.rgb_to_grayscale(real_clip)
    est_im_gray = tf.image.rgb_to_grayscale(est_clip)

    shape = tf.shape(real_clip_gray)
    reflect = tf.fill(shape, thres)  

    compa2 = tf.less(real_clip_gray, reflect)  
    compa3 = tf.less(est_im_gray, reflect)  
    compa = tf.to_float(compa2 & compa3) 

    kernel = np.array([
        [-1, -1, 0, 0, 0, -1, -1],
        [-1, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, -1],
        [-1, -1, 0, 0, 0, -1, -1]]).astype(np.float32).reshape(7, 7, 1)
    compa_erosion = tf.nn.erosion2d(compa, kernel, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                    padding="SAME")  
    compa_rgb = tf.tile(compa_erosion, [1, 1, 1, 3])
    loss_rec_sum = tf.reduce_sum(tf.multiply(tf.square(est_clip - real_clip - compensateI), compa_rgb), axis=[1, 2, 3])
    compa_sum = tf.reduce_sum(compa_rgb, axis=[1, 2, 3])
    loss_rec = tf.reduce_mean(tf.divide(loss_rec_sum, compa_sum))

    compa_sum_batch = tf.reduce_sum(compa_sum)
    loss_rec_sum_ = tf.reduce_mean(loss_rec_sum)

    return loss_rec, compa_sum_batch, loss_rec_sum_
