import os, gc
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import control_flow_ops
import time, argparse
import hubbard_input_data
starttime = time.localtime(time.time())
gc.enable()

Us = [4,5,6,8,9,10,12,14,16,20]
tempMins = [.18,.22,.29,.32,.38,.30,.29,.26,.22,.18]
tempMaxs = [.22,.27,.34,.37,.42,.35,.34,.31,.27,.22]

tempRange = False

for U in [8]
    for i in range(10):
        run_num = i + 1
        run_str = 'run' + str(run_num) + '_U' + str(U) + '/'

        if not os.path.exists(run_str):
            os.makedirs(run_str)


        tempMin = .18
        tempMax = .22

        try:
            data_file = 'Hubbard Data/N4x4x4_L200_U' + str(U) + '_Mu0/*.stream'
        except:
            data_file = '/home/kchng/Quantum Machine Learning/N4x4x4_L200_U' + str(U) + '_Mu0/*.stream'
            print('khatami_cluster')

        if tempRange == True:
            mnist = hubbard_input_data.dataAndLabels(data_file, tempMin=tempMin, tempMax=tempMax)
        else:
            mnist = hubbard_input_data.dataAndLabels(data_file)

        # Parameters
        learning_rate = 0.1
        training_epochs = 1*1E2
        batch_size = 10
        layer_trials = [[200,100,10,5]]

        side_squared = len(mnist.test.images[0])
        side = int(np.sqrt(side_squared))

        def layer_batch_norm(x, n_out, phase_train):
            beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
            gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

            beta = tf.get_variable("beta", [n_out], initializer=beta_init)
            gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
            def mean_var_with_update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            mean, var = control_flow_ops.cond(phase_train,
                mean_var_with_update,
                lambda: (ema_mean, ema_var))

            reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
            normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
                beta, gamma, 1e-3, True)
            return tf.reshape(normed, [-1, n_out])

        def layer(input, weight_shape, bias_shape, phase_train):
            weight_init = tf.random_normal_initializer(stddev=(1.0/weight_shape[0])**0.5)
            bias_init = tf.constant_initializer(value=0)
            W = tf.get_variable("W", weight_shape,
                                initializer=weight_init)
            b = tf.get_variable("b", bias_shape,
                                initializer=bias_init)
            logits = tf.matmul(input, W) + b
            return tf.nn.sigmoid(layer_batch_norm(logits, weight_shape[1], phase_train))

        def encoder(x, n_code, phase_train):
            with tf.variable_scope("encoder"):
                with tf.variable_scope("hidden_1"):
                    hidden_1 = layer(x, [side_squared, n_encoder_hidden_1], [n_encoder_hidden_1], phase_train)

                with tf.variable_scope("hidden_2"):
                    hidden_2 = layer(hidden_1, [n_encoder_hidden_1, n_encoder_hidden_2], [n_encoder_hidden_2], phase_train)

                with tf.variable_scope("hidden_3"):
                    hidden_3 = layer(hidden_2, [n_encoder_hidden_2, n_encoder_hidden_3], [n_encoder_hidden_3], phase_train)

                with tf.variable_scope("code"):
                    code = layer(hidden_3, [n_encoder_hidden_3, n_code], [n_code], phase_train)

            return hidden_1, hidden_2, hidden_3, code

        def decoder(code, n_code, phase_train):
            with tf.variable_scope("decoder"):
                with tf.variable_scope("hidden_1"):
                    hidden_1 = layer(code, [n_code, n_decoder_hidden_1], [n_decoder_hidden_1], phase_train)

                with tf.variable_scope("hidden_2"):
                    hidden_2 = layer(hidden_1, [n_decoder_hidden_1, n_decoder_hidden_2], [n_decoder_hidden_2], phase_train)

                with tf.variable_scope("hidden_3"):
                    hidden_3 = layer(hidden_2, [n_decoder_hidden_2, n_decoder_hidden_3], [n_decoder_hidden_3], phase_train)

                with tf.variable_scope("output"):
                    output = layer(hidden_3, [n_decoder_hidden_3, side_squared], [side_squared], phase_train)

            return hidden_1, hidden_2, hidden_3, output

        def loss(output, x):
            with tf.variable_scope("training"):
                l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(output, x)), 1))
                train_loss = tf.reduce_mean(l2)
                train_summary_op = tf.scalar_summary("train_cost", train_loss)
                return train_loss, train_summary_op

        def training(cost, global_step):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
                use_locking=False, name='Adam')
            train_op = optimizer.minimize(cost, global_step=global_step)
            return train_op

        def image_summary(label, tensor):
            tensor_reshaped = tf.reshape(tensor, [-1, side, side, 1])
            return tf.image_summary(label, tensor_reshaped)
            # return tf.image_summary(label,tensor)

        def evaluate(output, x):
            with tf.variable_scope("validation"):
                in_im_op = image_summary("input_image", x)
                out_im_op = image_summary("output_image", output)
                l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(output, x, name="val_diff")), 1))
                val_loss = tf.reduce_mean(l2)
                val_summary_op = tf.scalar_summary("val_cost", val_loss)
                return val_loss, in_im_op, out_im_op, val_summary_op


        if __name__ == '__main__':
            for L1, L2, L3, L4 in layer_trials:
                n_encoder_hidden_1 = L1
                n_encoder_hidden_2 = L2
                n_encoder_hidden_3 = L3
                n_code = L4
                n_decoder_hidden_1 = L3
                n_decoder_hidden_2 = L2
                n_decoder_hidden_3 = L1

                display_step = 1

                # how it saves the file name
                # AfterL1_[n_encoder_hidden_1]_[n_encoder_hidden_2]_[n_encoder_hidden_3]_[batch_size]_[learning_rate]_[training_epochs].txt
                filename_p1 = str(n_encoder_hidden_1)+'_'+str(n_encoder_hidden_2)+'_'+str(n_encoder_hidden_3)+'_'
                filename_p2 = str(n_code)+'_'+str(batch_size)+'_'+str(learning_rate)+'_'+str(int(training_epochs))
                filename = filename_p1 + filename_p2


                with tf.Graph().as_default():
                    with tf.variable_scope("autoencoder_model"):
                        # create placeholders
                        x = tf.placeholder("float", [None, side_squared])
                        phase_train = tf.placeholder(tf.bool)

                        # create model
                        encode_1, encode_2, encode_3, code = encoder(x, int(n_code), phase_train)
                        decode_1, decode_2, decode_3, output = decoder(code, int(n_code), phase_train)

                        # create loss function
                        cost, train_summary_op = loss(output, x)

                        # create training steps
                        global_step = tf.Variable(0, name='global_step', trainable=False)
                        train_op = training(cost, global_step)
                        eval_op, in_im_op, out_im_op, val_summary_op = evaluate(output, x)

                        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

                        # create graphs for tensorboard
                        train_writer = tf.train.SummaryWriter("hubbard_autoencoder_logs/" + filename + '/', graph=sess.graph)
                        val_writer = tf.train.SummaryWriter("hubbard_autoencoder_logs/" + filename + "/", graph=sess.graph)


                        # initialize all variables so things run smoothly
                        init_op = tf.initialize_all_variables()
                        sess.run(init_op)
                        saver = tf.train.Saver(max_to_keep=200)

                        # Training cycle
                        for epoch in range(int(training_epochs)):
                            avg_cost = 0.
                            total_batch = int(mnist.train.num_examples/batch_size)
                            # Loop over all batches
                            for i in range(total_batch):
                                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                                # Fit training using batch data
                                _, new_cost, train_summary = sess.run([train_op, cost, train_summary_op], feed_dict={x: minibatch_x, phase_train: True})
                                train_writer.add_summary(train_summary, sess.run(global_step))
                                # Compute average loss
                                avg_cost += new_cost/total_batch
                            # Display logs per epoch step
                            if epoch % display_step == 0:
                                print( "Epoch:", '%04d' % (epoch+1), "of %d " %training_epochs , "cost =", "{:.9f}".format(avg_cost))

                                # add training_summary to tensorboard
                                train_writer.add_summary(train_summary, sess.run(global_step))

                                validation_loss, in_im, out_im, val_summary = sess.run([eval_op, in_im_op, out_im_op, val_summary_op], feed_dict={x: mnist.validation.images, phase_train: False})
                                # add validation_loss summary to tensorboard
                                val_writer.add_summary(in_im, sess.run(global_step))
                                val_writer.add_summary(out_im, sess.run(global_step))
                                val_writer.add_summary(val_summary, sess.run(global_step))
                                print("Validation Loss:", validation_loss)

                                # create model checkpoint saver
                                if not os.path.exists("hubbard_autoencoder_logs/" + filename):
                                    os.makedirs("hubbard_autoencoder_logs/" + filename)

                                saver.save(sess, "hubbard_autoencoder_logs/" + filename + "/model-checkpoint-" + '%04d' % (epoch+1), global_step=global_step)
                                saver.export_meta_graph('hubbard_meta_graphs/' + filename + '/autoencoder.meta')

                        print( "Optimization Finished!")
                        test_loss = sess.run(eval_op, feed_dict={x: mnist.test.images, phase_train: False})
                        print("Test Loss:", test_loss)

                        sess = tf.InteractiveSession()
                        tf.initialize_all_variables().run()

                        # grab saved graph of NN to grab specific layers to eval
                        tf.train.import_meta_graph('hubbard_meta_graphs/' + filename + '/autoencoder.meta')
                        meta_graph_thing = tf.get_default_graph()

                        # used to figure out what each layer is called
                        # tensor_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

                        # creating tensors of each layer used for evaluation
                        encoder_L1W = meta_graph_thing.get_tensor_by_name('autoencoder_model/encoder/hidden_1/W/read:0')
                        encoder_L1b = meta_graph_thing.get_tensor_by_name('autoencoder_model/encoder/hidden_1/b/read:0')
                        encoder_L2W = meta_graph_thing.get_tensor_by_name('autoencoder_model/encoder/hidden_2/W/read:0')
                        encoder_L2b = meta_graph_thing.get_tensor_by_name('autoencoder_model/encoder/hidden_2/b/read:0')
                        encoder_L3W = meta_graph_thing.get_tensor_by_name('autoencoder_model/encoder/hidden_3/W/read:0')
                        encoder_L3b = meta_graph_thing.get_tensor_by_name('autoencoder_model/encoder/hidden_3/b/read:0')
                        encoder_L4W = meta_graph_thing.get_tensor_by_name('autoencoder_model/encoder/code/W/read:0')
                        encoder_L4b = meta_graph_thing.get_tensor_by_name('autoencoder_model/encoder/code/b/read:0')

                        if not os.path.exists(run_str + "layer_tensors/" + filename):
                            os.makedirs(run_str + "layer_tensors/" + filename)

                        L1W = tf.Print(encoder_L1W, [encoder_L1W]).eval()
                        L2W = tf.Print(encoder_L2W, [encoder_L2W]).eval()
                        L3W = tf.Print(encoder_L3W, [encoder_L3W]).eval()
                        L4W = tf.Print(encoder_L4W, [encoder_L4W]).eval()

                        L1b = tf.Print(encoder_L1b, [encoder_L1b]).eval()
                        L2b = tf.Print(encoder_L2b, [encoder_L2b]).eval()
                        L3b = tf.Print(encoder_L3b, [encoder_L3b]).eval()
                        L4b = tf.Print(encoder_L4b, [encoder_L4b]).eval()

                        np.savetxt(run_str + 'layer_tensors/' + filename + '/' + 'L1W.txt', L1W, delimiter='', fmt='%10.5f')
                        np.savetxt(run_str + 'layer_tensors/' + filename + '/' + 'L2W.txt', L2W, delimiter='', fmt='%10.5f')
                        np.savetxt(run_str + 'layer_tensors/' + filename + '/' + 'L3W.txt', L3W, delimiter='', fmt='%10.5f')
                        np.savetxt(run_str + 'layer_tensors/' + filename + '/' + 'L4W.txt', L4W, delimiter='', fmt='%10.5f')

                        np.savetxt(run_str + 'layer_tensors/' + filename + '/' + 'L1b.txt', L1b, delimiter='', fmt='%10.5f')
                        np.savetxt(run_str + 'layer_tensors/' + filename + '/' + 'L2b.txt', L2b, delimiter='', fmt='%10.5f')
                        np.savetxt(run_str + 'layer_tensors/' + filename + '/' + 'L3b.txt', L3b, delimiter='', fmt='%10.5f')
                        np.savetxt(run_str + 'layer_tensors/' + filename + '/' + 'L4b.txt', L4b, delimiter='', fmt='%10.5f')

                        def save_data():
                            # grab all images/labels into one array to evaluate layers on
                            all_images = []
                            all_labels = []
                            for i in range(len(mnist.test.images)):
                                all_images.append(mnist.test.images[i])
                                all_labels.append(mnist.test.labels[i])
                            for i in range(len(mnist.validation.images)):
                                all_images.append(mnist.validation.images[i])
                                all_labels.append(mnist.validation.labels[i])
                            for i in range(len(mnist.train.images)):
                                all_images.append(mnist.train.images[i])
                                all_labels.append(mnist.train.labels[i])

                            final_l1 = []
                            final_l2 = []
                            final_l3 = []
                            final_l4 = []

                            # evaulation loop, probably could be made faster if all in tensorflow
                            for i in range(len(all_images)):
                                # print( '\n\n\n' ,float(i)*100. / len(all_images), '%', '\n\n\n')
                                try:
                                    image = tf.cast(all_images[i], tf.float32)
                                    image = tf.reshape(image, [1,side_squared])
                                    After_l1 = tf.add(tf.matmul(image, encoder_L1W), encoder_L1b)
                                    After_l2 = tf.add(tf.matmul(After_l1, encoder_L2W), encoder_L2b)
                                    After_l3 = tf.add(tf.matmul(After_l2, encoder_L3W), encoder_L3b)
                                    After_l4 = tf.add(tf.matmul(After_l3, encoder_L4W), encoder_L4b)

                                    After_l1 = np.append(tf.Print(After_l1, [After_l1]).eval()[0], all_images[i])
                                    After_l2 = np.append(tf.Print(After_l2, [After_l2]).eval()[0], all_images[i])
                                    After_l3 = np.append(tf.Print(After_l3, [After_l3]).eval()[0], all_images[i])
                                    After_l4 = np.append(tf.Print(After_l4, [After_l4]).eval()[0], all_images[i])

                                    final_l1.append(After_l1)
                                    final_l2.append(After_l2)
                                    final_l3.append(After_l3)
                                    final_l4.append(After_l4)
                                except:
                                    print( 'something failed')
                            costs = [avg_cost, validation_loss, test_loss]

                            # saved data to file after having gone through NN up to each layer
                            if not os.path.exists(run_str + 'reduced_data/'+ filename):
                                os.makedirs(run_str + 'reduced_data/'+ filename)

                            np.savetxt(run_str + 'reduced_data/' + filename + '/AfterL1_' + filename + '.txt', final_l1, delimiter=' ')
                            np.savetxt(run_str + 'reduced_data/' + filename + '/AfterL2_' + filename + '.txt', final_l2, delimiter=' ')
                            np.savetxt(run_str + 'reduced_data/' + filename + '/AfterL3_' + filename + '.txt', final_l3, delimiter=' ')
                            np.savetxt(run_str + 'reduced_data/' + filename + '/AfterL4_' + filename + '.txt', final_l4, delimiter=' ')
                            np.savetxt(run_str + 'reduced_data/' + filename + '/Costs_'   + filename + '.txt', costs, delimiter=' ', header='avg_cost, validation_loss, test_loss')
            # print( L1,L2,L3,L4)

        # os.system("python bottlenecks.py")
        os.system("rm -rf hubbard_autoencoder_logs/")
        os.system("rm -rf hubbard_meta_graphs/")


        endtime = time.localtime(time.time())
        print( time.asctime(starttime))
        print( time.asctime(endtime))
