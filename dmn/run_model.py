import time
import os
import tensorflow as tf
from dmn_model import Config
from dmn_model import DmnModel


NUM_RUNS = 1
CONFIG_TRAIN = Config()
BEST_LOSS = float('inf')
with tf.variable_scope('DMN') as scope:
    MODEL = DmnModel(CONFIG_TRAIN)
for run in range(NUM_RUNS):

    print('Starting run', run)

    print('==> initializing variables')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        if Config.debug:
            session=tf.python.debug.LocalCLIDebugWrapperSession(session)

        sum_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)

        session.run(init)

        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        print('==> restoring weights')
        saver.restore(session, 'weights/task' +
                      str(MODEL.config.babi_id) + '.weights')

        print('==> starting training')
        for epoch in range(CONFIG_TRAIN.max_epoch):
            print('Epoch {}'.format(epoch))
            start = time.time()
            train_loss, train_accuracy = MODEL.run_epoch(
                session, MODEL.train, epoch, train_writer,
                train_op=MODEL.train_step)
            valid_loss, valid_accuracy = MODEL.run_epoch(session, MODEL.valid)
            print('Training loss: {}'.format(train_loss))
            print('Validation loss: {}'.format(valid_loss))
            print('Training accuracy: {}'.format(train_accuracy))
            print('Vaildation accuracy: {}'.format(valid_accuracy))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < BEST_LOSS:
                    print('Saving weights')
                    BEST_LOSS = best_val_loss
                    best_val_accuracy = valid_accuracy
                    saver.save(session, 'weights/task' +
                               str(MODEL.config.babi_id) + '.weights')

            # anneal
            if train_loss > prev_epoch_loss * MODEL.config.anneal_threshold:
                MODEL.config.lr /= MODEL.config.anneal_by
                print('annealed lr to %f' % MODEL.config.lr)

            prev_epoch_loss = train_loss

            if epoch - best_val_epoch > CONFIG_TRAIN.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))

        print('Best validation accuracy:', best_val_accuracy)
