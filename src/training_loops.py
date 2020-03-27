

def regular_training_loop(i):
    # training: do not train first epoch, to see random weights behaviour
    start_time = time.time()
    array_train_cost = []
    if i != 0:
        for train_batch in train_batch_streamer:
            tf_start = time.time()
            _, train_cost = sess.run([train_step, cost],
                                        feed_dict={x: train_batch['X'],
                                                y_: train_batch['Y'],
                                                lr: tmp_learning_rate,
                                                is_train: update_on_train})
            array_train_cost.append(train_cost)

    # validation
    array_val_cost = []
    for val_batch in val_batch_streamer:
        val_cost = sess.run([cost],
                            feed_dict={x: val_batch['X'],
                                        y_: val_batch['Y'],
                                        is_train: False})
        array_val_cost.append(val_cost)

    # Keep track of average loss of the epoch
    train_cost = np.mean(array_train_cost)
    val_cost = np.mean(array_val_cost)
    epoch_time = time.time() - start_time
    fy = open(model_folder + 'train_log.tsv', 'a')
    fy.write('%d\t%g\t%g\t%gs\t%g\n' % (i+1, train_cost, val_cost, epoch_time, tmp_learning_rate))
    fy.close()

    # Decrease the learning rate after not improving in the validation set
    if config['patience'] and k_patience >= config['patience']:
        print('Changing learning rate!')
        tmp_learning_rate = tmp_learning_rate / 2
        print(tmp_learning_rate)
        k_patience = 0

    # Early stopping: keep the best model in validation set
    if val_cost >= cost_best_model:
        k_patience += 1
        print('Epoch %d, train cost %g, val cost %g,'
                'epoch-time %gs, lr %g, time-stamp %s' %
                (i+1, train_cost, val_cost, epoch_time, tmp_learning_rate,
                str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))))

    else:
        # save model weights to disk
        save_path = saver.save(sess, model_folder)
        print('Epoch %d, train cost %g, val cost %g, '
                'epoch-time %gs, lr %g, time-stamp %s - [BEST MODEL]'
                ' saved in: %s' %
                (i+1, train_cost, val_cost, epoch_time,tmp_learning_rate,
                str(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())), save_path))
        cost_best_model = val_cost


def aversarial_training_loop(data):
    # training: do not train first epoch, to see random weights behaviour
    i, train_batch_streamer, sess, train_step, cost, t_cost, d_cost
    start_time = time.time()
    array_train_cost = []
    array_train_t_cost = []
    array_train_d_cost = []
    if i != 0:
        for train_batch in train_batch_streamer:
            tf_start = time.time()
            _, train_cost, train_t_cost, train_d_cost = sess.run([train_step, cost, t_cost, d_cost],
                                        feed_dict={x: train_batch['X'], y_: train_batch['Y'], d_: train_batch['D'], lr: tmp_learning_rate, is_train: True})
            array_train_cost.append(train_cost)
            array_train_t_cost.append(train_t_cost)
            array_train_d_cost.append(train_d_cost)

    # validation
    array_val_cost = []
    array_val_t_cost = []
    array_val_d_cost = []
    for val_batch in val_batch_streamer:
        val_cost, val_t_cost, val_d_cost = sess.run([cost, t_cost, d_cost],
                            feed_dict={x: val_batch['X'], y_: val_batch['Y'], d_: val_batch['D'], is_train: False})
        array_val_cost.append(val_cost)
        array_val_t_cost.append(val_t_cost)
        array_val_d_cost.append(val_d_cost)

    # Keep track of average loss of the epoch
    train_cost = np.mean(array_train_cost)
    train_t_cost = np.mean(array_train_t_cost)
    train_d_cost = np.mean(array_train_d_cost)

    val_cost = np.mean(array_val_cost)
    val_t_cost = np.mean(array_val_t_cost)
    val_d_cost = np.mean(array_val_d_cost)
    epoch_time = time.time() - start_time
    fy = open(model_folder + 'train_log.tsv', 'a')
    fy.write('%d\t%g\t%g\t%g\t%g\t%g\t%g\t%gs\t%g\n' % (i+1, train_cost, train_t_cost, train_d_cost, val_cost, val_t_cost, val_d_cost, epoch_time, tmp_learning_rate))
    fy.close()

    # Decrease the learning rate after not improving in the validation set
    if config['patience'] and k_patience >= config['patience']:
        print('Changing learning rate!')
        tmp_learning_rate = tmp_learning_rate / 2
        print(tmp_learning_rate)
        k_patience = 0

    # Early stopping: keep the best model in validation set
    if val_cost >= cost_best_model:
        k_patience += 1
        print('Epoch %d, train cost %g, train task cost %g, train discriminator cost %g, '
                'val cost %g, val task cost %g, val discriminator cost %g,'
                'epoch-time %gs, lr %g, time-stamp %s' %
                (i+1, train_cost, train_t_cost, train_d_cost, val_cost, val_t_cost, val_d_cost, epoch_time, tmp_learning_rate,
                str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))))

    else:
        # save model weights to disk
        save_path = saver.save(sess, model_folder)
        print('Epoch %d, train cost %g, train task cost %g, train discriminator cost %g, '
                'val cost %g, val task cost %g, val discriminator cost %g,'
                'epoch-time %gs, lr %g, time-stamp %s - [BEST MODEL]'
                ' saved in: %s' %
                (i+1, train_cost, train_t_cost, train_d_cost, val_cost, val_t_cost, val_d_cost, epoch_time,tmp_learning_rate,
                str(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())), save_path))
        cost_best_model = val_cost
