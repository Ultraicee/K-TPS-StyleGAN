import time
from collections import namedtuple
from utils import linear_sample, read_images,decoder

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
    return not_initialized_vars


def train_z(params, left_ims, right_ims, Gs, batch_idx, result_path, max_idx):
    # solve dlatents
    latents = np.zeros([params.data_size, Gs.input_shape[1]])
    latents_in = tf.constant(latents)
    dlatents = Gs.components.mapping.get_output_for(latents_in, None)
    # build tf graph
    with tf.variable_scope(tf.get_variable_scope()):
        left = tf.constant(left_ims, dtype=tf.float32)
        right = tf.constant(right_ims, dtype=tf.float32)
        z_f = tf.Variable(dlatents[:, 0, :], dtype=tf.float32)
        compensate_f = tf.constant(params.compensate_disp, dtype=tf.float32)

        z_input = tf.Variable(dlatents[:, 0, :], dtype=tf.float32, name='contr_val')
        compensate_disp = tf.Variable(compensate_f, dtype=tf.float32)

        update1 = tf.assign(z_input, z_f)
        update2 = tf.assign(compensate_disp, compensate_f)

        z_input_tile0 = tf.reshape(z_input, [params.data_size, 1, Gs.input_shape[1]])
        z_input_tile = tf.tile(z_input_tile0, [1, 14, 1])
        out_expr = Gs.components.synthesis.get_output_for(z_input_tile, truncation_psi=0.7,
                                                          randomize_noise=False)
        linear_interpolator = linear_sample.LinearInterpolator(params)  # initialize linear interpolator
        disp = decoder.decoder_forward1(z_input, linear_interpolator.sz_params, out_expr, compensate_disp * 10)
        right_est = linear_interpolator.interpolate(left, disp)

        if params.dataset_name == 'invivo':
            compensateI = tf.Variable(4.3, dtype=tf.float32, name='contr_val')
            loss_rec, compa_sum, loss_rec_sum = linear_sample.compute_rec_loss_per(right_est, right, compensateI, left,
                                                                     linear_interpolator.sz_params, 160.0)
        else:
            compensateI = tf.Variable(-2.0, dtype=tf.float32, name='contr_val')
            loss_rec, compa_sum, loss_rec_sum = linear_sample.compute_rec_loss(right_est, right, compensateI, left,
                                                                 linear_interpolator.sz_params, 160.0)

        disp_i1 = tf.slice(disp, [0, params.cutTop + 200, params.cutLeft + 200, 0], [-1, 55, 54, -1],
                           name='r_clip')
        disp_i2 = tf.slice(disp, [0, params.cutTop + 200, params.cutLeft + 201, 0], [-1, 55, 54, -1],
                           name='r_clip')
        disp_i3 = tf.slice(disp, [0, params.cutTop + 200, params.cutLeft + 200, 0], [-1, 54, 55, -1],
                           name='r_clip')
        disp_i4 = tf.slice(disp, [0, params.cutTop + 201, params.cutLeft + 200, 0], [-1, 54, 55, -1],
                           name='r_clip')

        loss_wt_norm = tf.multiply(1e-2, tf.reduce_sum(tf.square(disp_i2 - disp_i1)), name='punishment')
        loss_wt_norm2 = tf.multiply(1e-2, tf.reduce_sum(tf.square(disp_i4 - disp_i3)), name='punishment2')

        loss = tf.add(loss_rec, 0 * loss_wt_norm, name='Total_loss')
        loss = tf.add(loss, 0 * loss_wt_norm2, name='Total_loss')

        disp_cut = disp[:, params.cutTop:(params.cutTop + 256), params.cutLeft:(params.cutLeft + 256), :]
        disp_cut = tf.pad(disp_cut, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        filter_ = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]).reshape(3, 3, 1, 1)
        filter_1 = tf.constant(filter_, dtype='float32')
        Laplace_img = tf.nn.conv2d(input=disp_cut, filter=filter_1, strides=[1, 1, 1, 1], padding='VALID')

        loss_smooth = 50 * tf.reduce_mean(tf.square(Laplace_img))  # +0.01*tf.reduce_sum(tf.abs(Laplace_img))##0.8
        loss = tf.add(loss, 0.0 * loss_smooth, name='Total_loss')

        loss_wt_norm = tf.multiply(0.1, tf.reduce_sum(tf.square(z_input - z_f)), name='punishment')
        loss = tf.add(loss, loss_wt_norm, name='Total_loss')

        train_op = optimize_op.minimize(loss, var_list=[z_input])  # z_input_tile

    # session
    tf.get_default_session().run(tf.variables_initializer(get_uninitialized_variables(tf.get_default_session())))
    disps = []
    disps_all = []
    w_trained_stylegan = []
    res_loss_all = []
    w_stylegan_process_all = []

    for idx in range(max_idx):
        ids = range(idx * params.data_size, (idx + 1) * params.data_size)
        cur_left, cur_right = read_images.read_stereo_images(source_img_path, ids)
        start_time = time.time()
        z_before = dlatents[:, 0, :].eval()
        z_val = 0.
        disp_val = np.zeros([params.data_size, params.height, params.width, 1])
        est_right_val = np.zeros([params.data_size, params.height, params.width, params.channel])
        res_loss = [[], []]
        disps_perimg = []
        w_stylegan_process = []
        for step in range(0, max_step):
            _, z_val, loss_rec_val, compa_sum_val, loss_rec_sum_val, loss_val, compensate_disp_val, disp_val, est_right_val = tf.get_default_session().run(
                [train_op, z_input, loss_rec, compa_sum, loss_rec_sum, loss, compensate_disp, disp, right_est],
                feed_dict={left: cur_left, right: cur_right}
            )
            res_loss[0].append(loss_rec_sum_val * params.data_size)
            res_loss[1].append(compa_sum_val)
            disps_perimg.append(disp_val)
            w_stylegan_process.append(z_val)
            if 0 == step % 10 or step + 1 == max_step:
                z_var_mean = np.mean(z_val - z_before)

                print(print_str.format(step, loss_rec_val, loss_val, z_var_mean))
                z_before = z_val
                if step >= 150:
                    break

        print('time spent {:8} '.format(time.time() - start_time))

        tf.get_default_session().run(update1)

        disps.append(disp_val)
        disps_all.append(disps_perimg)
        w_trained_stylegan.append(z_val)
        res_loss_all.append(res_loss)
        w_stylegan_process_all.append(w_stylegan_process)

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
                          cutRight=90,  # invivo
                          dataset_name='invivo',
                          # cutTop=16,
                          # cutBottom=16,
                          # cutLeft=70,
                          # cutRight=34,  #phantom
                          # dataset_name = 'phantom',
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
        train_z(params, left_ims, right_ims, Gs, idx, 'gt_z_per/', max_idx)
