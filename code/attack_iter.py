from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging

import numpy as np
import random
import pandas as pd
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize

import tensorflow as tf

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
from nets.mobilenet import mobilenet_v2
from nets.nasnet import pnasnet

slim = tf.contrib.slim


tf.flags.DEFINE_string('checkpoint_path', './model', 'Path to checkpoint.')

tf.flags.DEFINE_string('input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_string('exp_name', '', 'Name of the experiment.')

tf.flags.DEFINE_float('max_epsilon', 32.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_float('prob', 0.4, 'probability of using diverse inputs.')

FLAGS = tf.flags.FLAGS

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'resnet_v2_152': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_152.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'mobilenet_v2_1.0': os.path.join(FLAGS.checkpoint_path, 'mobilenet_v2_1.0_224.ckpt'),
    'pnasnet-5_mobile': os.path.join(FLAGS.checkpoint_path, 'pnasnet-5_mobile_model_modify.ckpt'),
    'resnet_v2_101': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt'),
}

def gkern(kernlen=21, nsig=3):
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

kernel = gkern(15, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)

def load_labels(file_name):
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['ImageId']+'.png': dev.iloc[i]['TrueLabel'] for i in range(len(dev))}
    return f2l


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imresize(imread(f, mode='RGB'), [FLAGS.image_height, FLAGS.image_width]).astype(np.float) / 255.0
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')

def graph(x, y, i, num_iter, alpha, x_max, x_min, beta, beta_, grad):

    momentum = FLAGS.momentum
    num_classes = 1001

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet_152, end_points_resnet_152 = resnet_v2.resnet_v2_152(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet_101, end_points_resnet_101 = resnet_v2.resnet_v2_101(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(pnasnet.pnasnet_mobile_arg_scope()):
        logits_pnasnet, end_points_pnasnet = pnasnet.build_pnasnet_mobile(
            input_diversity(x), num_classes=num_classes, is_training=False, scope='pnasnet_mobile')

    with slim.arg_scope(mobilenet_v2.training_scope()):
        logits_mobilenet_v2, end_points_mobilenet_v2 = mobilenet_v2.mobilenet(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_ens_adv_res_v2, end_points_ens_adv_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(x), num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

    logits = logits_v3 * beta[0] + logits_v4 * beta[1] + logits_res_v2 * beta[2] + logits_ens3_adv_v3 * beta[3] \
            + logits_ens4_adv_v3 * beta[4] + logits_ens_adv_res_v2 * beta[5] + logits_resnet_101 * beta[6] \
            + logits_resnet_152 * beta[7] + logits_mobilenet_v2 * beta[8] + logits_pnasnet * beta[9]
    auxlogits = end_points_v3['AuxLogits'] * beta_[0] + end_points_v4['AuxLogits'] * beta_[1]+ end_points_res_v2['AuxLogits'] * beta_[2] \
            + end_points_ens3_adv_v3['AuxLogits'] * beta_[3] + end_points_ens4_adv_v3['AuxLogits'] * beta_[4] + end_points_ens_adv_res_v2['AuxLogits'] * beta_[5]
    cross_entropy = tf.losses.softmax_cross_entropy(y,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(y,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=0.4)
    noise = tf.gradients(cross_entropy, x)[0]
    noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, y, i, num_iter, alpha, x_max, x_min, beta, beta_, noise


def stop(x, y, i, num_iter, alpha, x_max, x_min, beta, beta_, grad):
    return tf.less(i, num_iter)


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)


def check_or_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def cal_diff(diff_images):
    diff_images = diff_images * 0.5 * 255
    diff_images = diff_images.reshape(-1, FLAGS.image_width * FLAGS.image_height * 3)
    L_infty = np.linalg.norm(diff_images, ord=np.inf, axis=1)
    L_1 = np.linalg.norm(diff_images, ord=1, axis=1)/(FLAGS.image_width * FLAGS.image_height * 3)
    diff_images = diff_images.reshape(-1, FLAGS.image_width * FLAGS.image_height, 3)
    L_2 = np.mean(np.sqrt(np.sum(diff_images ** 2, axis=2)), axis=1)
    return list(L_infty), list(L_1), list(L_2)


def main(_):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    f2l = load_labels(os.path.join(FLAGS.input_dir, '..', 'dev_dataset.csv'))

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.exp_name)
    check_or_create_dir(output_dir)

    # Logger
    tf.logging.set_verbosity(tf.logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    handler = logging.FileHandler(os.path.join(FLAGS.output_dir, "attack_iter_%s.log" % FLAGS.exp_name), mode='w')
    handler.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        beta_input = [tf.placeholder(tf.float32) for i in range(10)]
        beta_input_ = [tf.placeholder(tf.float32) for i in range(6)]
        grad_input = tf.placeholder(tf.float32, shape=batch_shape)
        num_iter_input = tf.placeholder(tf.int32)
        alpha_input = tf.placeholder(tf.float32, shape=())
        labels_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        y = tf.one_hot(labels_input, num_classes)

        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        i = tf.constant(0)
        x_adv, _, _, _, _, _, _, _, _, grad_adv = tf.while_loop(stop, graph, [x_input, y, i, num_iter_input, alpha_input, x_max, x_min, beta_input, beta_input_, grad_input])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))
        s9 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        s10 = tf.train.Saver(slim.get_model_variables(scope='MobilenetV2/'))
        s11 = tf.train.Saver(slim.get_model_variables(scope='pnasnet_mobile'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            s3.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            s4.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            s5.restore(sess, model_checkpoint_map['inception_v4'])
            s6.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            s8.restore(sess, model_checkpoint_map['resnet_v2_101'])
            s9.restore(sess, model_checkpoint_map['resnet_v2_152'])
            s10.restore(sess, model_checkpoint_map['mobilenet_v2_1.0'])
            s11.restore(sess, model_checkpoint_map['pnasnet-5_mobile'])
            
            eps = 2.0 * FLAGS.max_epsilon / 255.0
            alpha_test = eps / FLAGS.num_iter
            alpha_train = eps / 16
            

            idx = 0
            L_infty = []
            L_1 = []
            L_2 = []
            start_time = time.time()
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                labels = [f2l[filename] for filename in filenames]
                adv_images = images.copy()
                grad_images = np.zeros(batch_shape)
                for i in range(FLAGS.num_iter):
                    train_index = random.sample(range(10), 6)
                    test_index = train_index.pop()

                    # meta train step
                    num_iter = 8
                    beta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    for j in train_index:
                        beta[j] = 1 / len(train_index)
                    beta_ = [0, 0, 0, 0, 0, 0]

                    auxlogits_num = np.sum(np.array(train_index) < 6)
                    if auxlogits_num != 0:
                        for j in train_index:
                            if j < 6:
                                beta_[j] = 1 / auxlogits_num

                    beta_dict = {beta_input[j]: beta[j] for j in range(10)}
                    beta_dict_ = {beta_input_[j]: beta_[j] for j in range(6)}
                    adv_images_temp, grad_images = sess.run([x_adv, grad_adv], feed_dict={x_input: adv_images, labels_input: labels, grad_input: grad_images, num_iter_input: num_iter, alpha_input: alpha_train, **beta_dict, **beta_dict_})

                    # meta test step
                    num_iter = 1
                    beta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    beta[test_index] = 1
                    beta_ = [0, 0, 0, 0, 0, 0]
                    if test_index < 6:
                        beta_[test_index] = 1
                    beta_dict = {beta_input[j]: beta[j] for j in range(10)}
                    beta_dict_ = {beta_input_[j]: beta_[j] for j in range(6)}
                    adv_images_temp2, grad_images = sess.run([x_adv, grad_adv], feed_dict={x_input: adv_images_temp, labels_input: labels, grad_input: grad_images, num_iter_input: num_iter, alpha_input: alpha_test, **beta_dict, **beta_dict_})
                    adv_images += adv_images_temp2 - adv_images_temp

                save_images(adv_images, filenames, output_dir)
                idx = idx + 1
                print("start the i={} attack".format(idx))
                diff_images = adv_images - images
                L_infty_, L_1_, L_2_ = cal_diff(diff_images)
                logger.debug("%s %s %s" % (L_infty_, L_1_, L_2_))
                L_infty += L_infty_
                L_1 += L_1_
                L_2 += L_2_

        print(time.time() - start_time)
        logger.info("[L_infty] MAX: %.4f AVG: %.4f\n[L_1] MAX: %.4f AVG: %.4f\n[L_2] MAX: %.4f AVG: %.4f\n" % (np.max(L_infty), np.mean(L_infty), np.max(L_1), np.mean(L_1), np.max(L_2), np.mean(L_2)))


if __name__ == '__main__':
    tf.app.run()
