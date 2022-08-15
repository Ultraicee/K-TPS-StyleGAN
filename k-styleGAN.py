import time
from collections import namedtuple
from utils import linear_sample, read_images, decoder

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import tensorflow as tf
import dnnlib.tflib as tflib
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_params = namedtuple("parameters",
                          'data_size,'
                          'mini_batch_size,'
                          'learning_rate,'
                          'total_epoch_num,'
                          'outputdir,'
                          'height,'
                          'width,'
                          'channel,'
                          'cutTop,'
                          'cutBottom,'
                          'cutLeft,'
                          'cutRight,'
                          'dataset_name,'
                          'cPointRow,'
                          'cPointCol,'
                          'compensate_disp,')

print_str = 'train_tps_Step：{:4} | Reonstruction loss is {:4} | Total loss is {:4} | z_var_mean is{:4}'


def get_uninitialized_variables(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    print([str(i.name) for i in not_initialized_vars])
    return not_initialized_vars


def train_z(params, left_ims, right_ims, Gs, batch_idx, result_path):
    # solve dlatents
    latents = np.zeros([params.data_size, Gs.input_shape[1]])
    latents_in = tf.constant(latents)
    dlatents = Gs.components.mapping.get_output_for(latents_in, None)
    # build tf graph
    with tf.variable_scope(tf.get_variable_scope()):
        left = tf.constant(left_ims, dtype=tf.float32)
        right = tf.constant(right_ims, dtype=tf.float32)
        z_input = tf.Variable(dlatents[:, 0, :], dtype=tf.float32, name='contr_val')

        # repeat 14 times
        z_input_tile0 = tf.reshape(z_input, [params.data_size, 1, Gs.input_shape[1]])
        z_input_tile = tf.tile(z_input_tile0, [1, 14, 1])
        out_expr = Gs.components.synthesis.get_output_for(z_input_tile, truncation_psi=0.7,
                                                          randomize_noise=False)
        compensate_disp = tf.Variable(params.compensate_disp, dtype=tf.float32, name='contr_val')
        linear_interpolator = linear_sample.LinearInterpolator(params)  # initialize linear interpolator
        disp = decoder.decoder_forward1(z_input, linear_interpolator.sz_params, out_expr, compensate_disp * 10)
        right_est = linear_interpolator.interpolate(left, disp)
        # compute loss of reconstruction
        if params.dataset_name == 'invivo':
            compensateI = tf.Variable(4.3, dtype=tf.float32, name='contr_val')
            loss_rec, compa_sum, loss_rec_sum = linear_sample.compute_rec_loss_per(right_est, right, compensateI, left,
                                                                     linear_interpolator.sz_params, 160.0)
        else:
            # other dataset is 'phantom' as paper tested
            compensateI = tf.Variable(-2.0, dtype=tf.float32, name='contr_val')
            loss_rec, compa_sum, loss_rec_sum = linear_sample.scompute_rec_loss(right_est, right, compensateI, left,
                                                                 linear_interpolator.sz_params, 160.0)
        disp_i1 = tf.slice(disp, [0, params.cutTop, params.cutLeft, 0], [-1, 256, 255, -1],
                           name='r_clip')
        disp_i2 = tf.slice(disp, [0, params.cutTop, params.cutLeft + 1, 0], [-1, 256, 255, -1],
                           name='r_clip')
        disp_i3 = tf.slice(disp, [0, params.cutTop, params.cutLeft, 0], [-1, 255, 256, -1],
                           name='r_clip')
        disp_i4 = tf.slice(disp, [0, params.cutTop + 1, params.cutLeft, 0], [-1, 255, 256, -1],
                           name='r_clip')

        loss_wt_norm = tf.multiply(1e-4, tf.reduce_sum(tf.square(disp_i2 - disp_i1)), name='punishment')
        loss_wt_norm2 = tf.multiply(1e-4, tf.reduce_sum(tf.square(disp_i4 - disp_i3)), name='punishment2')
        loss = tf.add(loss_rec, loss_wt_norm, name='Total_loss')
        loss = tf.add(loss, loss_wt_norm2, name='Total_loss')

        train_op = optimize_op.minimize(loss, var_list=[z_input, compensate_disp])  # z_input_tile

    # run session
    tf.get_default_session().run(tf.variables_initializer(get_uninitialized_variables(tf.get_default_session())))
    start_time = time.time()
    loss_rec_temp = 0.

    z_before = dlatents[:, 0, :].eval()
    z_val = 0.
    disp_val = np.zeros([params.data_size, params.height, params.width, 1])
    est_right_val = np.zeros([params.data_size, params.height, params.width, params.channel])
    step = 0
    res_loss = [[], []]
    for step in range(0, max_step):
        _, z_val, loss_rec_val, compa_sum_val, loss_rec_sum_val, loss_val, compensate_disp_val, disp_val, est_right_val = tf.get_default_session().run(
            [train_op, z_input, loss_rec, compa_sum, loss_rec_sum, loss, compensate_disp, disp, right_est]
        )
        res_loss[0].append(loss_rec_sum_val * params.data_size)
        res_loss[1].append(compa_sum_val)

        if 0 == step % 10 or step + 1 == max_step:
            z_before = z_val
            loss_var_interp = np.abs(loss_val - loss_rec_temp)
            loss_rec_temp = loss_val
            z_var_mean = np.mean(z_val - z_before)
            print(print_str.format(step, loss_rec_val, loss_val, z_var_mean))
            if loss_var_interp < 1e-3 or step >= 200:
                break
    z = z_val
    print('time spent {:8} '.format(time.time() - start_time))
    np.save(os.path.join(result_path, 'W_disp_batch' + str(batch_idx) + '.npy'), disp_val)
    np.save(os.path.join(result_path, 'G_est_right_real_batch' + str(batch_idx) + '.npy'), est_right_val)


if __name__ == '__main__':
    params = model_params(data_size=50,
                          mini_batch_size=50,
                          learning_rate=5e-2,
                          total_epoch_num=np.int32(20000),
                          outputdir=r'output',
                          height=288,
                          width=360,
                          channel=3,

                          cutTop=32,
                          cutBottom=0,
                          cutLeft=14,
                          cutRight=90, 
                          dataset_name='invivo',
                          cPointRow=5,
                          cPointCol=6,
                          compensate_disp=8.52
                          )
    
    # modify the path of your own datasets and pre-trained styleGAN model
    source_img_path = 'datasets/invivo1_rect/'
    model_path = 'results/00002-sgan-MNdatasets-1gpu/network-snapshot-001170.pkl'
    max_idx = 12 
    # load pre-trained G
    tflib.init_tf()
    url = os.path.abspath(model_path)
    # input zeros to solve latent vector w
    with open(url, 'rb') as f:
        _, _, Gs = pickle.load(f)
     
    max_step = np.int32(params.total_epoch_num)
    # learning policy
    learning_rate_init = np.float32(params.learning_rate)
    optimize_op = tf.train.AdamOptimizer(learning_rate_init)  # ,0.9,0.999,1e-08
    optimize_op_compensate_disp = tf.train.AdamOptimizer(5e-1)  # ,0.9,0.999,1e-08
    for idx in range(max_idx):
        ids = range(idx * params.data_size, (idx + 1) * params.data_size)
        left_ims, right_ims = read_images.read_stereo_images(source_img_path, ids)
        train_z(params, left_ims, right_ims, Gs, idx, result_path = 'gt_z/')
