import os
import random
import numpy as np
import logging
import tensorflow as tf
from scipy.misc import imread, imresize, imsave
import pandas as pd
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2, vgg
from nets.mobilenet import mobilenet_v2
from nets.nasnet import pnasnet, nasnet

slim = tf.contrib.slim

tf.flags.DEFINE_string('checkpoint_path', './model', 'Path to checkpoint.')

tf.flags.DEFINE_string('input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_string('exp_name', '', 'Name of the experiment.')

FLAGS = tf.flags.FLAGS

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
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'mobilenet_v2_1.4': os.path.join(FLAGS.checkpoint_path, 'mobilenet_v2_1.4_224_modify.ckpt'),
    'nasnet-a_mobile': os.path.join(FLAGS.checkpoint_path, 'nasnet-a_mobile_model_modify.ckpt'),
}


def load_labels(file_name):
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['ImageId']+'.png': dev.iloc[i]['TrueLabel'] for i in range(len(dev))}
    return f2l


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
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


def main(_):
    f2l = load_labels(os.path.join(FLAGS.input_dir, '..', 'dev_dataset.csv'))
    input_dir = os.path.join(FLAGS.output_dir, FLAGS.exp_name)

    batch_shape = [50, 299, 299, 3]
    num_classes = 1001
    tf.logging.set_verbosity(tf.logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    
    handler = logging.FileHandler(os.path.join(FLAGS.output_dir, "simple_eval_%s.log" % FLAGS.exp_name), mode='w')
    handler.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_v4, end_points_v4 = inception_v4.inception_v4(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ens_adv_res_v2, end_points_ens_adv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet_152, end_points_resnet_152 = resnet_v2.resnet_v2_152(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet_101, end_points_resnet_101 = resnet_v2.resnet_v2_101(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(mobilenet_v2.training_scope()):
            logits_mobilenet_v2, end_points_mobilenet_v2 = mobilenet_v2.mobilenet(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(mobilenet_v2.training_scope()):
            logits_mobilenet_v2_14, end_points_mobilenet_v2_14 = mobilenet_v2.mobilenet_v2_140(
                x_input, num_classes=num_classes, is_training=False, scope='MobilenetV2_1.4')

        with slim.arg_scope(pnasnet.pnasnet_mobile_arg_scope()):
            logits_pnasnet, end_points_pnasnet = pnasnet.build_pnasnet_mobile(
                x_input, num_classes=num_classes, is_training=False, scope='pnasnet_mobile')

        with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
            logits_nasnet, end_points_nasnet = nasnet.build_nasnet_mobile(
                x_input, num_classes=num_classes, is_training=False, scope='nasnet_mobile')

        pred_v3 = tf.argmax(end_points_v3['Predictions'], 1)
        pred_adv_v3 = tf.argmax(end_points_adv_v3['Predictions'], 1)
        pred_ens3_adv_v3 = tf.argmax(end_points_ens3_adv_v3['Predictions'], 1)
        pred_ens4_adv_v3 = tf.argmax(end_points_ens4_adv_v3['Predictions'], 1)
        pred_v4 = tf.argmax(end_points_v4['Predictions'], 1)
        pred_res_v2 = tf.argmax(end_points_res_v2['Predictions'], 1)
        pred_ens_adv_res_v2 = tf.argmax(end_points_ens_adv_res_v2['Predictions'], 1)
        pred_resnet_152 = tf.argmax(end_points_resnet_152['predictions'], 1)
        pred_resnet_101 = tf.argmax(end_points_resnet_101['predictions'], 1)
        pred_mobilenet_v2_14 = tf.argmax(end_points_mobilenet_v2_14['Predictions'], 1)
        pred_mobilenet_v2 = tf.argmax(end_points_mobilenet_v2['Predictions'], 1)
        pred_pnasnet = tf.argmax(end_points_pnasnet['Predictions'], 1)
        pred_nasnet = tf.argmax(end_points_nasnet['Predictions'], 1)


        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))
        s9 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        s10 = tf.train.Saver(slim.get_model_variables(scope='MobilenetV2/'))
        s11 = tf.train.Saver(slim.get_model_variables(scope='pnasnet_mobile'))
        s12 = tf.train.Saver(slim.get_model_variables(scope='nasnet_mobile'))
        s13 = tf.train.Saver(slim.get_model_variables(scope='MobilenetV2_1.4'))
    


        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            s2.restore(sess, model_checkpoint_map['adv_inception_v3'])
            s3.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            s4.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            s5.restore(sess, model_checkpoint_map['inception_v4'])
            s6.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            s8.restore(sess, model_checkpoint_map['resnet_v2_101'])
            s9.restore(sess, model_checkpoint_map['resnet_v2_152'])
            s10.restore(sess, model_checkpoint_map['mobilenet_v2_1.0'])
            s11.restore(sess, model_checkpoint_map['pnasnet-5_mobile'])
            s12.restore(sess, model_checkpoint_map['nasnet-a_mobile'])
            s13.restore(sess, model_checkpoint_map['mobilenet_v2_1.4'])


            model_name = ['inception_v3', 'inception_v4', 'inception_resnet_v2',
                          'resnet_v2_152', 'ens3_adv_inception_v3', 'ens4_adv_inception_v3',
                          'ens_adv_inception_resnet_v2', 'adv_inception_v3', 'resnet_v2_101',
                          'mobilenet_v2_1.0', 'pnasnet-5_mobile', 'nasnet-a_mobile', 'mobilenet_v2_1.4']
            success_count = np.zeros(len(model_name))

            idx = 0
            for filenames, images in load_images(input_dir, batch_shape):
                idx += 1
                print("start the i={} eval".format(idx))
                v3, adv_v3, ens3_adv_v3, ens4_adv_v3, v4, res_v2, ens_adv_res_v2, resnet_101, resnet_152, mobile_v2, pnasnet_mobile, nasnet_mobile, mobile_v2_14 = sess.run(
                    (pred_v3, pred_adv_v3, pred_ens3_adv_v3, pred_ens4_adv_v3, pred_v4, pred_res_v2,
                     pred_ens_adv_res_v2, pred_resnet_101, pred_resnet_152, pred_mobilenet_v2, pred_pnasnet, 
                     pred_nasnet, pred_mobilenet_v2_14), feed_dict={x_input: images})

                for filename, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13 in zip(filenames, v3, adv_v3, ens3_adv_v3,
                                                                    ens4_adv_v3, v4, res_v2, ens_adv_res_v2,
                                                                    resnet_152, resnet_101, mobile_v2, pnasnet_mobile, nasnet_mobile, mobile_v2_14):
                    label = f2l[filename]
                    l = [l1, l5, l6, l8, l3, l4, l7, l2, l9, l10, l11, l12, l13]
                    for i in range(len(model_name)):
                        if l[i] != label:
                            success_count[i] += 1

            for i in range(len(model_name)):
                logger.info("Attack Success Rate for {0} : {1:.1f}%".format(model_name[i], success_count[i] / 1000. * 100))



if __name__ == '__main__':
    tf.app.run()
    