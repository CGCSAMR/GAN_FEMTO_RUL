#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 22:31:28 2018

@author: ubuntusamuel
"""

#import os
import data_pros
import scoring
#from skimage import data
import tensorflow as tf
#import random as rdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

import csv
import os

"""
*******************************************************************************
                             CARGA DE ARCHIVOS  
*******************************************************************************
"""
#Train_dir: Directorio de ubicación de los datos de salida 
#Test_dir: Dierectorio de ubicación de los datos de prueba
#Train_list: Nombre de los rodamiento de entrenamiento
#Test_list: Nombre de los rodamiento de prueba
#Train_file_list: Nombre de los archivos asociados a cada rodameinto de entrenamiento
#Test_file_list: Nombre de los archivos asociados a cada rodamieno de prueba
train_dir, test_dir, train_list, test_list, train_file_list, test_file_list = data_pros.data_pth()

#Index_table_tr: Numeros de ejemplos de entrenameinto
#Unit_table_tr: Numero de rodamiento asociado a la muestra de entrenamiento 
#Number_unit_tr: Numero de muestra dentro del rodamiento de entrenameinto selecionado
index_table_tr, unit_table_tr, number_unit_tr = data_pros.index_dir(train_file_list)

#Index_table_ts: Numeros de ejemplos de prueba
#Unit_table_ts: Numero de rodamiento asociado a la muestra de prueba 
#Number_unit_ts: Numero de muestra dentro del rodamiento de prueba selecionado
index_table_ts, unit_table_ts, number_unit_ts = data_pros.index_dir(test_file_list)


rul = [5730, 339, 1610, 1460, 7570, 7530, 1390, 3090, 1290, 580, 820]
"""
Ejemplo de imagen cargada       
image_ex = data.load(train_dir + '/' + train_list[0] + '/' + train_file_list[0][0], as_grey = True)
"""

"""
*******************************************************************************
                                    MRGAN 
*******************************************************************************
"""


#mode = "train"
mode = "eval"
#mode = "none"

rnn = "lstm"
#rnn = None

lstm_mode = "train"
#lstm_mode = "eval"
#lstm_mode = "none"

start = False

max_seq = 512

h_dim = 128
lam1 = 1e-2
lam2 = 1e-2

g_1 = tf.Graph()

with g_1.as_default():
#Entradas: tf_x = Imagenes 64,64 para entrenar
#          Z = Ruido para generar las imagenes    
    tf_x = 2*tf.placeholder(tf.float32, [None, 64,64,1]) / 255.-1.
    Z = tf.placeholder(tf.float32, shape=[None,1,1,100])


    def sample_Z(m, n):
        """
        sample_Z: int int -> tensor
        crea el vator de ruido para crear la imagen    
        """
        return np.random.uniform(-1., 1., size=[m, 1, 1, n])
        #return np.ramdom.normal(loc=0.0, scale=1.0, size=[m, 1, 1, n])

    def encoder(tf_x, res):
        """
        encoder: tensor bool -> tensor
        red que toma las imegenes y las pasa a la dimension del ruido
        """
        with tf.variable_scope('Encoder', reuse = res):
            batch_norm_params = {'decay': 0.999,
                                 'epsilon': 0.001,
                                 'scope': 'batch_norm'}
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d],
                                                kernel_size=[4, 4],
                                                stride=[2, 2],
                                                activation_fn=leakyrelu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                normalizer_params=batch_norm_params,
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.00004, scope='l2_decay')):

            # images: 64 x 64 x 3
                layer1 = tf.contrib.layers.conv2d(inputs=tf_x,
                                                  num_outputs=16,
                                                  normalizer_fn=None,
                                                  biases_initializer=None,
                                                  scope='layer1')
            # layer1: 32 x 32 x (64)
                layer2 = tf.contrib.layers.conv2d(inputs=layer1,
                                                  num_outputs=32,
                                                  scope='layer2')
            # layer2: 16 x 16 x (128)
                layer3 = tf.contrib.layers.conv2d(inputs=layer2,
                                                  num_outputs=64,
                                                  scope='layer3')
            # layer2: 8 x 8 x (256)
                layer4 = tf.contrib.layers.conv2d(inputs=layer3,
                                                  num_outputs=128,
                                                  scope='layer4')
            # layer2: 4 x 4 x (512)
        
                layer5 = tf.contrib.layers.conv2d(inputs=layer4,
                                                  num_outputs=100,
                                                  stride=[1, 1],
                                                  padding='VALID',
                                                  normalizer_fn=None,
                                                  normalizer_params=None,
                                                  activation_fn=tf.nn.tanh,
                                                  scope='layer5')

                discriminator_logits = layer5

                return discriminator_logits


    def generator(z, res):
        """
        generator: tensor bool -> tensor
        Red generadora de imagenes a partir de ruido
        """
        with tf.variable_scope('Generator', reuse = res):
            batch_norm_params = {'decay': 0.999,
                                 'epsilon': 0.001,
                                 'is_training': True,
                                 'scope': 'batch_norm'}
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d_transpose],
                                                kernel_size=[4, 4],
                                                stride=[2, 2],
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                normalizer_params=batch_norm_params,
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.00004, scope='l2_decay')):
            
                layer0 = tf.contrib.layers.fully_connected(inputs = z,
                                                           num_outputs=100,
                                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                                           normalizer_params=batch_norm_params,
                                                           weights_regularizer=tf.contrib.layers.l2_regularizer(0.00004, scope='l2_decay'),
                                                           scope='layer0')
                #Input a 1 x 1 x 100 tensor 
                layer1 = tf.contrib.layers.conv2d_transpose(inputs=layer0,
                                                            num_outputs=128,
                                                            padding='VALID',
                                                            scope='layer1')
                # layer1: 4 x 4 x (512)
                layer2 = tf.contrib.layers.conv2d_transpose(inputs=layer1,
                                                            num_outputs=64,
                                                            scope='layer2')
                # layer2: 8 x 8 x (256)
        
                layer3 = tf.contrib.layers.conv2d_transpose(inputs=layer2,
                                                            num_outputs=32,
                                                            scope='layer3')
                # layer2: 16 x 16 x (128)
        
                layer4 = tf.contrib.layers.conv2d_transpose(inputs=layer3,
                                                            num_outputs=16,
                                                            scope='layer4')
                # layer2: 32 x 32 x (64)
                layer5 = tf.contrib.layers.conv2d_transpose(inputs=layer4,
                                                            num_outputs=1,
                                                            normalizer_fn=None,
                                                            biases_initializer=None,
                                                            activation_fn=tf.tanh,
                                                            scope='layer5')
                # output = layer5: 28 x 28 x 1
                generated_images = layer5
            
                return generated_images


    def discriminator(tf_x, res):
        """
        discriminator: tensor bool -> float
        Red disrimindaora, toma imagenes y saca una porbabilidad
        """
        with tf.variable_scope('Discriminator', reuse = res):
            batch_norm_params = {'decay': 0.999,
                                 'epsilon': 0.001,
                                 'scope': 'batch_norm'}
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d],
                                                kernel_size=[4, 4],
                                                stride=[2, 2],
                                                activation_fn=leakyrelu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                normalizer_params=batch_norm_params,
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.00004, scope='l2_decay')):
                
                # images: 64 x 64 x 3
                layer1 = tf.contrib.layers.conv2d(inputs=tf_x,
                                                  num_outputs=16,
                                                  normalizer_fn=None,
                                                  biases_initializer=None,
                                                  scope='layer1')
                # layer1: 32 x 32 x (64)
                layer2 = tf.contrib.layers.conv2d(inputs=layer1,
                                                  num_outputs=32,
                                                  scope='layer2')
                # layer2: 16 x 16 x (128)
                layer3 = tf.contrib.layers.conv2d(inputs=layer2,
                                                  num_outputs=64,
                                                  scope='layer3')
                # layer2: 8 x 8 x (256)
                layer4 = tf.contrib.layers.conv2d(inputs=layer3,
                                                  num_outputs=128,
                                                  scope='layer4')
                # layer2: 4 x 4 x (512)
                
                layer5 = tf.contrib.layers.conv2d(inputs=layer4,
                                                  num_outputs=1,
                                                  stride=[1, 1],
                                                  padding='VALID',
                                                  normalizer_fn=None,
                                                  normalizer_params=None,
                                                  activation_fn=None,
                                                  scope='layer5')
                
                discriminator_logits = layer5
                
                return layer4, discriminator_logits
        
    def leakyrelu(x, leaky_weight=0.2, name=None):
        """
        leakyrelu: float float bool -> float
        funcion de   activacion leakyrelu
        """
        return tf.maximum(x, leaky_weight*x)

    def plot(samples):
        """
        plot tensor -> imagen
        Grafica las imagenes generadas
        """
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(64, 64),cmap='Greys_r')

        return fig

    def log(x):
        """
        log: foat -> float
        calcual logarimos cercanos a cero
        """
        return tf.log(x+1e-8)

    # Z -> G
    G_sample = generator(Z, None)
    # tf_x -> E
    E = encoder(tf_x, None)
    # tf_x -> E -> G
    G_sample_reg = generator(encoder(tf_x, True), True)


    # tf_x -> D
    D_out, D_real = discriminator(tf_x, None)
    # Z -> G -> D
    _, D_fake = discriminator(G_sample, True)
    # tf_x -> E -> Z -> G -> D
    _, D_reg = discriminator(G_sample_reg, True)

    # Distancia entre imagenes reales e imagenes generadas a partir de ellas    
    mse = tf.reduce_sum((tf_x - G_sample_reg)**2, 1)

    # Perdidas varias
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
    D_loss = D_loss_real + D_loss_fake
    E_loss = tf.reduce_mean(lam1 * mse - lam2 * tf.nn.sigmoid_cross_entropy_with_logits(logits=D_reg, labels=tf.ones_like(D_reg)))
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))+ E_loss

    #D_loss = -tf.reduce_mean(log(D_real) + log(1 - D_fake))
    #E_loss = tf.reduce_mean(lam1 * mse + lam2 * log(D_reg))
    #G_loss = -tf.reduce_mean(log(D_fake)) + E_loss

    # Varialbes a ajustar
    
    t_vars = tf.trainable_variables()
      
    D_vars = [var for var in t_vars if 'Discriminator' in var.name]
    G_vars = [var for var in t_vars if 'Generator' in var.name]
    E_vars = [var for var in t_vars if 'Encoder' in var.name]

    # Optimizacion de funcinoes de costo
    E_solver = (tf.train.AdamOptimizer(learning_rate=1e-3)
                .minimize(E_loss, var_list=E_vars))
    D_solver = (tf.train.AdamOptimizer(learning_rate=1e-3)
                .minimize(D_loss, var_list=D_vars))
    G_solver = (tf.train.AdamOptimizer(learning_rate=1e-3)
                .minimize(G_loss, var_list=G_vars))

    mb_size = 64
    z_dim = 100

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
    
        if not os.path.exists('mrgan2_model/'):
            os.makedirs('mrgan2_model/')
        
        if mode == "train":
            
            if not os.path.exists('outmrgan2/'):
                os.makedirs('outmrgan2/')

            i = 0
            itl = []
            GL = []
            DL = []
            EL = []
        
            maximo = 1
            post5 = 0
            post10 = 0
            post20 = 0
        
            if start:
                saver.restore(sess, "mrgan2_model/mrgan_start.ckpt")  

            index = 0
            for it in range(100000):
                index, X_mb = data_pros.random_selection(index, mb_size, index_table_tr, unit_table_tr, number_unit_tr, train_dir, train_list, train_file_list)
                _, D_loss_curr = sess.run(
                        [D_solver, D_loss],
                        feed_dict={tf_x: X_mb, Z: sample_Z(mb_size, z_dim)}
                        )

                _, G_loss_curr = sess.run(
                        [G_solver, G_loss],
                        feed_dict={tf_x: X_mb, Z: sample_Z(mb_size, z_dim)}
                        )

                _, E_loss_curr = sess.run(
                        [E_solver, E_loss],
                        feed_dict={tf_x: X_mb, Z: sample_Z(mb_size, z_dim)}
                        )

                itl.append(it)
                GL.append(G_loss_curr)
                DL.append(D_loss_curr)
                EL.append(E_loss_curr)
            
                if it >= 500:
                    
                    if (D_loss_curr >= maximo):
                        save_path = saver.save(sess, "mrgan2_model/mrgan_max.ckpt")
                        maximo = D_loss_curr
                        post5 = it + 5
                        post10 = it + 1000
                        post20 = it + 1500
                
                        print('max found at step:')
                        print(it)
                    
                        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, z_dim)})
                        
                        fig = plot(samples)
                        plt.savefig('outmrgan2/max.png', bbox_inches='tight')
            
                    elif it == post5:
                        save_path = saver.save(sess, "mrgan2_model/mrgan_post5.ckpt")
                        
                        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, z_dim)})
                        
                        fig = plot(samples)
                        plt.savefig('outmrgan2/post5.png', bbox_inches='tight')
                
                    elif it == post10:
                        save_path = saver.save(sess, "mrgan2_model/mrgan_post10.ckpt") 
                
                        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, z_dim)})
                    
                        fig = plot(samples)
                        plt.savefig('outmrgan2/post10.png', bbox_inches='tight')
                
                    elif it == post20:
                        save_path = saver.save(sess, "mrgan2_model/mrgan_post20.ckpt")                
                    
                        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, z_dim)})
                
                        fig = plot(samples)
                        plt.savefig('outmrgan2/post20.png', bbox_inches='tight')
                    

                if it % 100 == 0:
                    print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; E_loss: {:.4}'
                          .format(it, D_loss_curr, G_loss_curr, E_loss_curr))

                    samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, z_dim)})
            
                    fig = plot(samples)
                    plt.savefig('outmrgan2/{}.png'
                                .format(str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)
            
                    if it == 0:
                            save_path = saver.save(sess, "mrgan2_model/mrgan_start.ckpt")
                    
                    save_path = saver.save(sess, "mrgan2_model/mrgan_end.ckpt")
    
            with open('mrgan2.csv', 'wb') as csvfile:
                cost = csv.writer(csvfile)
                cost.writerow(itl)
                cost.writerow(GL)
                cost.writerow(DL)
                cost.writerow(EL)
                
        if mode == "eval":
            saver.restore(sess, "mrgan2_model/mrgan_end.ckpt")
            #saver.restore(sess, "saved_model/mrgan_max.ckpt")
            #saver.restore(sess, "daved_model/mrgan_post5.ckpt")
            #saver.restore(sess, "saved_model/mrgan_post10.ckpt")
            #saver.restore(sess, "saved_model/mrgan_post20.ckpt")
            
            rul_tr, images_tr = data_pros.selection(unit_table_tr, number_unit_tr, 
                                                    train_dir, train_list, train_file_list)
            gan_out = []
            
            for i in range(len(images_tr)):
                unit = sess.run(D_out, feed_dict = {tf_x: images_tr[i].reshape((1,64,64,1))})
                gan_out.append(unit.reshape(4*4*128))
            
            #gan_out = sess.run(D_out, feed_dict = {tf_x: images_tr})
            
            rul_ts, images_ts = data_pros.selection(unit_table_ts, number_unit_ts, 
                                                    test_dir, test_list, test_file_list)
            gan_out_t = []
            fail_cycle_t = []
            
            for i in range(len(images_ts)):
                unit = sess.run(D_out, feed_dict = {tf_x: images_ts[i].reshape((1,64,64,1))})
                gan_out_t.append(unit.reshape(4*4*128))
                
            #gan_out_t = sess.run(D_out, feed_dict = {tf_x: images_ts})
            
            lstm_gan_tr, seq_tr = data_pros.eval_data(gan_out, number_unit_tr, max_seq)
            lstm_gan_ts, seq_ts = data_pros.eval_data(gan_out_t, number_unit_ts, max_seq)
            #R_early = 125
            
            #max_seq, data_out, rul_out, seq_out = entrada.lstm_data(gan_out,fail_cycle, R_early)
            
            #data_out_t, seg_t = entrada.lstm_test(gan_out_t, max_seq)
            
            #data_out_e, seg_e = entrada.lstm_test(gan_out, max_seq)
        sess.close()
        
"""
*******************************************************************************
                                    RNN
*******************************************************************************                                    
"""        
        
if rnn == "lstm":
    
    g_2 = tf.Graph()
    
    with g_1.as_default():   
    # ==========
    #   MODEL
    # ==========

    # Parameters
        learning_rate = 0.01
        training_steps = 50000
        batch_size = 1024
        display_step = 100
        seq_max_len = max_seq # Sequence max length
    # Network Parameters
        
        n_hidden = 256 # hidden layer num of features
        n_classes = 1 # linear sequence or not

        index = 0
        #trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
        #testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

    # tf Graph input
        x = tf.placeholder("float", [None, seq_max_len, 4*4*128])
        y = tf.placeholder("float", [None, n_classes])
    # A placeholder for indicating each sequence length
        seqlen = tf.placeholder(tf.int64, [None])

    # Define weights
        weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
        biases = {'out': tf.Variable(tf.random_normal([n_classes]))}


        def dynamicRNN(x, seqlen, weights, biases):
            
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
            # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.unstack(x, seq_max_len, 1)

            # Define a lstm cell with tensorflow
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

            # Get lstm cell output, providing 'sequence_length' will perform dynamic
            # calculation.
            outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

            # When performing dynamic calculation, we must retrieve the last
            # dynamically computed output, i.e., if a sequence length is 10, we need
            # to retrieve the 10th output.
            # However TensorFlow doesn't support advanced indexing yet, so we build
            # a custom op that for each sample in batch size, get its length and
            # get the corresponding relevant output.

            # 'outputs' is a list of output at every timestep, we pack them in a Tensor
            # and change back dimension to [batch_size, n_step, n_input]
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])

            # Hack to build the indexing and retrieve the right output.
            batch_size = tf.shape(outputs, out_type= tf.int64)[0]
            # Start indices for each sample
            index = tf.range(0, batch_size, dtype= tf.int64) * seq_max_len + (seqlen - 1)
            # Indexing
            outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

            X_o = tf.nn.dropout(outputs, 0.9)
            # Linear activation, using outputs computed above
            return tf.matmul(X_o, weights['out']) + biases['out']

        pred = dynamicRNN(x, seqlen, weights, biases)

        # Define loss and optimizer
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        
        cost = tf.reduce_sum(tf.pow((pred-y),2))/(batch_size)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        # Evaluate model
        #correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)
            
            if lstm_mode == "train":
            
                final_index = 0

                save_path = saver.save(sess, "saved_model/lstm_start.ckpt")
            
                for step in range(1, training_steps + 1):
                    #batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
                    index, seq_out, rul_out, seq_len = data_pros.random_lstm(index, mb_size, seq_max_len,
                                                                          rul_tr, index_table_tr, number_unit_tr, 
                                                                          gan_out, train_dir, train_list, train_file_list)
                    #X_mb, y_mb, seq_mb, final_index, final_batch = entrada.lstm_batch(rm_data_out, rm_rul_out, rm_seq_out, batch_size, final_index)
                
                    #if final_index == len(rm_rul_out):
                    #    rm_data_out, rm_rul_out, rm_seq_out = entrada.mezclar(data_out, rul_out, seq_out)
                
                    # Run optimization op (backprop)
                    sess.run(optimizer, feed_dict={x: seq_out, y: rul_out, seqlen: seq_len})
                    if step % display_step == 0 or step == 1:
                        # Calculate batch accuracy & loss
                        loss = sess.run(cost, feed_dict={x: seq_out, y: rul_out, seqlen: seq_len})
                    
                        print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss))
                
                print("Optimization Finished!")
                save_path = saver.save(sess, "saved_model/lstm_end.ckpt")
            
            
            elif lstm_mode == "eval":
                saver.restore(sess, "saved_model/lstm_end.ckpt")
                
                prediccion_e = sess.run(pred, feed_dict = {x: lstm_gan_tr, seqlen: seq_tr})
                
                prediccion_t = sess.run(pred, feed_dict = {x: lstm_gan_ts, seqlen: seq_ts})
                
                score_e = scoring.scoring_fun(np.zeros(len(prediccion_e)), prediccion_e) 
                
                rmse_e = scoring.rmse(np.zeros(len(prediccion_e)), prediccion_e)
                
                score_t = scoring.scoring_fun(rul, prediccion_t)
                 
                rmse_t = scoring.rmse(rul, prediccion_t)
                
                print("Score train")
                print(score_e)
                
                print("RMSE train")
                print(rmse_e)
                
                print("Score test")
                print(score_t)
                
                print("RMSE test")
                print(rmse_t)
            # Calculate accuracy
            #test_data = testset.data
            #test_label = testset.labels
            #test_seqlen = testset.seqlen
            #print("Testing Accuracy:", \
            #sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))
            sess.close()
        