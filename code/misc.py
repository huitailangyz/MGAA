import tensorflow as tf
import tensorflow.contrib.slim as slim

def name_in_checkpoint(name):
  return name.replace('MobilenetV2', 'MobilenetV2_1.4')
  # return 'nasnet_mobile/' + name

def rename_var(ckpt_path, new_ckpt_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(ckpt_path):
            var = tf.contrib.framework.load_variable(ckpt_path, var_name)
            new_var_name = name_in_checkpoint(var_name)
            var = tf.Variable(var, name=new_var_name)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, new_ckpt_path)

old_path = "../model/mobilenet_v2_1.4_224.ckpt"
new_path = "../model/mobilenet_v2_1.4_224_test.ckpt"
rename_var(old_path, new_path)