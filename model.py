# coding = utf-8

import tensorflow as tf
import numpy as np
import tensorflow.nn as nn

slim = tf.contrib.slim


def arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME') as arg_sc:
            return arg_sc


def img_scale(x, scale):
    weight = x.get_shape()[1].value
    height = x.get_shape()[2].value

    try:
        out = tf.image.resize_nearest_neighbor(x, size=(weight*scale, height*scale))
    except:
        out = tf.image.resize_images(x, size=[weight*scale, height*scale])
    return out

def instance_norm(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

def relu(x):
    return tf.nn.relu(x)

def weight_variable(shape,n):
    initial = lambda: tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=n)

def bias_variable(shape,n):
    initial = lambda: tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=n)
    
def conv2d(x,w,s,p,n):
    return tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding=p, name=n)

def ConvLstm(X0, H0, C0, chi, chxi, cho, reuse, name, is_train=True):
    with tf.variable_scope(name, reuse=reuse) as vs:
        wxi = weight_variable([3,3, chxi, cho], 'WLSix')
        x0 = tf.pad(X0, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        ix = conv2d(x0,wxi,1,'VALID','LSix')
        whi = weight_variable([3,3, chi, cho], 'WLSih')
        H0 = tf.pad(H0, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        ih = conv2d(H0,whi,1,'VALID','LSih')
        bi = bias_variable([cho], 'BLSi')
        i = tf.sigmoid(ih + ix + bi)
        
        wxf = weight_variable([3,3, chxi, cho], 'WLSfx')
        fx = conv2d(x0,wxf,1,'VALID','LSfx')
        whf = weight_variable([3,3, chi, cho], 'WLSfh')
        fh = conv2d(H0,whf,1,'VALID','LSfh')
        bf = bias_variable([cho], 'BLSf')
        f = tf.sigmoid(fh + fx + bf)
        
        wxo = weight_variable([3,3, chxi, cho], 'WLSox')
        ox = conv2d(x0,wxo,1,'VALID','LSox')
        who = weight_variable([3,3, chi, cho], 'WLSoh')
        oh = conv2d(H0,who,1,'VALID','LSoh')
        bo = bias_variable([cho], 'BLSo')
        o = tf.sigmoid(oh + ox + bo)
        
        wxc = weight_variable([3,3, chxi, cho], 'WLScx')
        cx = conv2d(x0,wxc,1,'VALID','LScx')
        whc = weight_variable([3,3, chi, cho], 'WLSch')
        ch = conv2d(H0,whc,1,'VALID','LSch')
        bc = bias_variable([cho], 'BLSc')
        C1 = f * C0 + i * tf.nn.tanh(cx + ch + bc)
        H1 = o * tf.nn.tanh(C1)
        
    return C1, H1

def Conv(fin, Hin, cho, reuse, name, is_train=True, activation_fn=nn.relu):
    with tf.variable_scope(name, reuse=reuse) as vs:
        fin = tf.concat([fin, Hin], axis=3)
        fin = tf.pad(fin, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        out = slim.conv2d(fin, cho, [3, 3], stride=1, padding='VALID', activation_fn=activation_fn, scope='_3_1')
    return out

def CConvLSTM(cho, X0, H0, C0, chi, chxi, reuse, name, is_train=True, activation_fn=nn.relu):
    with tf.variable_scope(name, reuse=reuse) as vs:
        out = Conv(X0, H0, cho, reuse, 'conv2d', is_train, activation_fn)+X0
        C1, H1 = ConvLstm(X0, H0, C0, chi, chxi, cho, reuse, 'ConvLSTM', is_train)
    return C1, H1, out

def SequentialBlock(imgs, height, kk, F0, X0, H0, C0, X1, H1, C1, X2, H2, C2, X3, H3, C3, X4, Out):
    with tf.variable_scope("LineStyle", reuse=tf.AUTO_REUSE) as vs:

        kkn = kk+8
        imgsn = imgs[:,8:,:,:]
        F0n = imgsn[:,0:8,:,:]
        F = tf.pad(F0, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='REFLECT')
        X0n = slim.conv2d(F, 32, [9, 9], stride=1, scope='head1')
        X0n = relu(instance_norm(X0n))

        X0n = slim.conv2d(X0n, 64, [3, 3], stride=1, scope='head2')
        X0n = instance_norm(X0n)

        X0n = slim.conv2d(X0n, 128, [3, 3], stride=1, scope='head3')
        X0n = instance_norm(X0n)

        c0, h0, x1 = CConvLSTM(128, X0, H0, C0, 128, 128, False, 'LB0', True)
        X1n = x1
        C0n = c0
        H0n = h0
        c1, h1, x2 = CConvLSTM(128, X1, H1, C1, 128, 128, False, 'LB2', True)
        X2n = x2
        C1n = c1
        H1n = h1
        c2, h2, x3 = CConvLSTM(128, X2, H2, C2, 128, 128, False, 'LB4', True)
        X3n = x3
        C2n = c2
        H2n = h2
        c3, h3, x4 = CConvLSTM(128, X3, H3, C3, 128, 128, False, 'LB6', True)
        X4n = x4
        C3n = c3
        H3n = h3       

        out4 = slim.conv2d(X4, 64, [3, 3], stride=1, scope='conv4')
        out4 = relu(instance_norm(out4))

        out4 = slim.conv2d(out4, 32, [3, 3], stride=1, scope='conv5')
        out4 = relu(instance_norm(out4))

        out = slim.conv2d(out4, 3, [9, 9], scope='conv6')
        out = tf.nn.tanh(instance_norm(out))

        out = (out + 1) * 127.5

        height1 = out.get_shape()[1].value 
        width1 = out.get_shape()[2].value  

        out = tf.image.crop_to_bounding_box(out, 4, 4, height1-8, width1-8)
        
        
        Outn = tf.concat([Out, out], axis=1)
    
    return imgsn, height, kkn, F0n, X0n, H0n, C0n, X1n, H1n, C1n, X2n, H2n, C2n, X3n, H3n, C3n, X4n, Outn

def cond(imgs, height, kk, F0, X0, H0, C0, X1, H1, C1, X2, H2, C2, X3, H3, C3, X4, Out):
    return kk < height

def SequentialStyle(imgs, reuse, name, is_train=True):
    with tf.variable_scope(name, reuse=reuse) as vs:
        h = imgs.get_shape()[1]
        w = imgs.get_shape()[2].value
        n = imgs.get_shape()[0].value
        c = imgs.get_shape()[3].value
        lines = tf.constant(8)
        cc = lines - tf.mod(h, lines) + 10*lines
        height = cc + h - lines 
        imgs = tf.pad(imgs, [[0, 0], [0, cc], [0, 0], [0, 0]], mode='REFLECT')
        
        kk = tf.constant(0,name='kk')
        F0 = imgs[:,kk:kk+lines,:,:]
        padd = 8
        
        X0 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="X0")
        H0 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="H0")
        C0 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="C0")
        X1 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="X1")
        H1 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="H1")
        C1 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="C1")
        X2 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="X2")
        H2 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="H2")
        C2 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="C2")
        X3 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="X3")
        H3 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="H3")
        C3 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="C3")
        
        X4 = tf.zeros(shape=[n, tf.to_int32((lines+padd)), tf.to_int32((w+padd)), 128], dtype=tf.float32, name="X4")
        Out = tf.zeros(shape=[n, lines, w, 3], dtype=tf.float32, name="Out")
        
        imgs, height, kk, F0, X0, H0, C0, X1, H1, C1, X2, H2, C2, X3, H3, C3, X4, Out = tf.while_loop(cond, SequentialBlock, [imgs, height, kk, F0, X0, H0, C0, X1, H1, C1, X2, H2, C2, X3, H3, C3, X4, Out], shape_invariants=[tf.TensorShape([n, None, w, 3]), height.get_shape(), kk.get_shape(), tf.TensorShape([n, None, w, 3]), tf.TensorShape([n, None, (w+padd), 128]), H0.get_shape(), C0.get_shape(), X1.get_shape(), H1.get_shape(), C1.get_shape(), tf.TensorShape([n, None, (w+padd), 128]), H2.get_shape(), C2.get_shape(), X3.get_shape(), H3.get_shape(), C3.get_shape(), X4.get_shape(), tf.TensorShape([n, None, w, 3])])
       
        out = tf.image.crop_to_bounding_box(Out, 10*lines, 0, h, w)
        
        variables = tf.contrib.framework.get_variables(vs)
    
    return out, variables

"""caculate the loss"""
import vgg_simple as vgg
import os


def styleloss(f1, f2, f3, f4):
    gen_f, _, style_f = tf.split(f1, 3, 0)
    size = tf.size(gen_f)
    style_loss = tf.nn.l2_loss(gram(gen_f) - gram(style_f))*2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f2, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f3, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f4, 3, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(style_f)) * 2 / tf.to_float(size)

    return style_loss

def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams
