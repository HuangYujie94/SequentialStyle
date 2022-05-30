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
        x0 = tf.pad(X0, [[0, 0], [1, 1], [1, 1], [0, 0]])
        ix = conv2d(x0,wxi,1,'VALID','LSix')
        whi = weight_variable([3,3, chi, cho], 'WLSih')
        H0 = tf.pad(H0, [[0, 0], [1, 1], [1, 1], [0, 0]])
        ih = conv2d(H0,whi,1,'VALID','LSih')
        wci = weight_variable(C0.shape[1:], 'WLSic')
        ic = wci*C0
        bi = bias_variable([cho], 'BLSi')
        i = tf.sigmoid(ic + ih + ix + bi)
        
        wxf = weight_variable([3,3, chxi, cho], 'WLSfx')
        fx = conv2d(x0,wxf,1,'VALID','LSfx')
        whf = weight_variable([3,3, chi, cho], 'WLSfh')
        fh = conv2d(H0,whf,1,'VALID','LSfh')
        wcf = weight_variable(C0.shape[1:], 'WLSfc')
        fc = wcf*C0
        bf = bias_variable([cho], 'BLSf')
        f = tf.sigmoid(fc + fh + fx + bf)
        
        wxo = weight_variable([3,3, chxi, cho], 'WLSox')
        ox = conv2d(x0,wxo,1,'VALID','LSox')
        who = weight_variable([3,3, chi, cho], 'WLSoh')
        oh = conv2d(H0,who,1,'VALID','LSoh')
        wco = weight_variable(C0.shape[1:], 'WLSoc')
        oc = wco*C0
        bo = bias_variable([cho], 'BLSo')
        o = tf.sigmoid(oc + oh + ox + bo)
        
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
        fin = tf.pad(fin, [[0, 0], [1, 1], [1, 1], [0, 0]])
        out = slim.conv2d(fin, cho, [3, 3], stride=1, padding='VALID', activation_fn=activation_fn, scope='_3_1')
    return out

def CConvLSTM(cho, X0, H0, C0, chi, chxi, reuse, name, is_train=True, activation_fn=nn.relu):
    with tf.variable_scope(name, reuse=reuse) as vs:
        out = Conv(X0, H0, cho, reuse, 'conv2d', is_train, activation_fn)
        C1, H1 = ConvLstm(X0, H0, C0, chi, chxi, cho, reuse, 'ConvLSTM', is_train)
    return C1, H1, out

def SequentialBlock(imgs, height, kk, F0, X0, H0, C0, X1, H1, C1, X2, H2, C2, X3, H3, C3, X4, H4, C4, X5, H5, C5, X6, H6, C6, X7, H7, C7, X8, H8, C8, X9, H9, C9, X10, H10, C10, X11, H11, C11, X12, Out):
    with tf.variable_scope("SequentialStyle", reuse=tf.AUTO_REUSE) as vs:

        kkn = kk+8
        imgsn = imgs[:,8:,:,:]
        F0n = imgsn[:,0:8,:,:]
        F = tf.pad(F0, [[0, 0], [1, 1], [1, 1], [0, 0]])
        X0n = slim.conv2d(F, 32, [3, 3], stride=1, padding='VALID', scope='head')
        
        c0, h0, x1 = CConvLSTM(32, X0, H0, C0, 32, 32, False, 'LB0', True)
        c1, h1, x2 = CConvLSTM(32, X1, H1, C1, 32, 32, False, 'LB1', True)
        c2, h2, x3 = CConvLSTM(32, X2, H2, C2, 32, 32, False, 'LB2', True)
        c3, h3, x4 = CConvLSTM(32, X3, H3, C3, 32, 32, False, 'LB3', True)
        X1n = x1
        X2n = x2
        X3n = x3
        X4n = x4
        C0n = c0
        H0n = h0
        C1n = c1
        H1n = h1
        C2n = c2
        H2n = h2
        C3n = c0
        H3n = h3
       
        c4, h4, x5 = CConvLSTM(64, X4, H4, C4, 64, 32, False, 'LB4', True)
        c5, h5, x6 = CConvLSTM(64, X5, H5, C5, 64, 64, False, 'LB5', True)
        c6, h6, x7 = CConvLSTM(64, X6, H6, C6, 64, 64, False, 'LB6', True)
        c7, h7, x8 = CConvLSTM(64, X7, H7, C7, 64, 64, False, 'LB7', True)
        X5n = x5
        X6n = x6
        X7n = x7
        X8n = x8
        C4n = c4
        H4n = h4
        C5n = c5
        H5n = h5
        C6n = c6
        H6n = h6
        C7n = c7
        H7n = h7
        
        c8, h8, x9 = CConvLSTM(128, X8, H8, C8, 128, 64, False, 'LB8', True)
        c9, h9, x10 = CConvLSTM(128, X9, H9, C9, 128, 128, False, 'LB9', True)
        c10, h10, x11 = CConvLSTM(128, X10, H10, C10, 128, 128, False, 'LB10', True)
        c11, h11, x12 = CConvLSTM(128, X11, H11, C11, 128, 128, False, 'LB11', True, activation_fn=nn.tanh)
        X9n = x9
        X10n = x10
        X11n = x11
        X12n = x12
        C8n = c8
        H8n = h8
        C9n = c9
        H9n = h9
        C10n = c10
        H10n = h10
        C11n = c11
        H11n = h11
        
        X12 = tf.pad(X12, [[0, 0], [1, 1], [1, 1], [0, 0]])
        Xout = slim.conv2d(X12, 3, [3, 3], stride=1, padding='VALID', activation_fn=nn.tanh, scope='out')
        Xout = (Xout + 1) * 127.5
        Outn = tf.concat([Out, Xout], axis=1)
    
    return imgsn, height, kkn, F0n, X0n, H0n, C0n, X1n, H1n, C1n, X2n, H2n, C2n, X3n, H3n, C3n, X4n, H4n, C4n, X5n, H5n, C5n, X6n, H6n, C6n, X7n, H7n, C7n, X8n, H8n, C8n, X9n, H9n, C9n, X10n, H10n, C10n, X11n, H11n, C11n, X12n, Outn


def cond(imgs, height, kk, F0, X0, H0, C0, X1, H1, C1, X2, H2, C2, X3, H3, C3, X4, Out):
    return kk < height

def SequentialStyle(imgs, reuse, name, is_train=True):
    with tf.variable_scope(name, reuse=reuse) as vs:
        h = imgs.get_shape()[1]
        w = imgs.get_shape()[2].value
        n = imgs.get_shape()[0].value
        c = imgs.get_shape()[3].value
        lines = tf.constant(8)
        cc = 1000
        height = cc + h - lines 
        imgs = tf.pad(imgs, [[0, 0], [0, cc], [0, 0], [0, 0]], mode='REFLECT')
        
        kk = tf.constant(0,name='kk')
        F0 = imgs[:,kk:kk+lines,:,:]
        
        X0 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="X0")
        H0 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="H0")
        C0 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="C0")
        X1 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="X1")
        H1 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="H1")
        C1 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="C1")
        X2 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="X2")
        H2 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="H2")
        C2 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="C2")
        X3 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="X3")
        H3 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="H3")
        C3 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="C3")
        
        X4 = tf.zeros(shape=[n, lines, w, 32], dtype=tf.float32, name="X4")
        H4 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="H4")
        C4 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="C4")
        X5 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="X5")
        H5 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="H5")
        C5 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="C5")
        X6 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="X6")
        H6 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="H6")
        C6 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="C6")
        X7 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="X7")
        H7 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="H7")
        C7 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="C7")
        
        X8 = tf.zeros(shape=[n, lines, w, 64], dtype=tf.float32, name="X8")
        H8 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="H8")
        C8 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="C8")
        X9 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="X9")
        H9 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="H9")
        C9 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="C9")
        X10 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="X10")
        H10 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="H10")
        C10 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="C10")
        X11 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="X11")
        H11 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="H11")
        C11 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="C11")
        
        X12 = tf.zeros(shape=[n, lines, w, 128], dtype=tf.float32, name="X12")
        Out = tf.zeros(shape=[n, lines, w, 3], dtype=tf.float32, name="Out")
        
        imgs, height, kk, F0, X0, H0, C0, X1, H1, C1, X2, H2, C2, X3, H3, C3, X4, H4, C4, X5, H5, C5, X6, H6, C6, X7, H7, C7, X8, H8, C8, X9, H9, C9, X10, H10, C10, X11, H11, C11, X12, Out = tf.while_loop(cond, SequentialBlock, [imgs, height, kk, F0, X0, H0, C0, X1, H1, C1, X2, H2, C2, X3, H3, C3, X4, H4, C4, X5, H5, C5, X6, H6, C6, X7, H7, C7, X8, H8, C8, X9, H9, C9, X10, H10, C10, X11, H11, C11, X12, Out], shape_invariants=[tf.TensorShape([n, None, w, 3]), height.get_shape(), kk.get_shape(), tf.TensorShape([n, None, w, 3]), tf.TensorShape([n, None, w, 32]), H0.get_shape(), C0.get_shape(), X1.get_shape(), H1.get_shape(), C1.get_shape(), X2.get_shape(), H2.get_shape(), C2.get_shape(), X3.get_shape(), H3.get_shape(), C3.get_shape(), X4.get_shape(), H4.get_shape(), C4.get_shape(), X5.get_shape(), H5.get_shape(), C5.get_shape(), X6.get_shape(), H6.get_shape(), C6.get_shape(), X7.get_shape(), H7.get_shape(), C7.get_shape(), X8.get_shape(), H8.get_shape(), C8.get_shape(), X9.get_shape(), H9.get_shape(), C9.get_shape(), X10.get_shape(), H10.get_shape(), C10.get_shape(), X11.get_shape(), H11.get_shape(), C11.get_shape(), X12.get_shape(), tf.TensorShape([n, None, w, 3])])
        
        out = tf.image.crop_to_bounding_box(Out, 14*lines, 0, h, w)
        
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
