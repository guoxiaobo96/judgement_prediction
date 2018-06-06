#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnnModelTest import TCNNConfig,TextCnn,CharLevelCNN,TestModel,TestCnnConv2,HierachyCnn,TestHierachyCnn
from prepareData import read_vocab,batch_iter, get_data, build_vocab,read_catagory,read_word2vec,get_data_with_vocab
import cnnModel

base_dir = '/home/guoxiaobo96/criminal_data'
data_dir=os.path.join(base_dir,'criminal/data.txt')
#train_dir = os.path.join(base_dir, 'data_train.txt')
#test_dir = os.path.join(base_dir, '2.txt')
#val_dir = os.path.join(base_dir, 'data_valid.txt')
vocab_dir = os.path.join(base_dir, 'vocab/data_model.txt')

save_dir = '/home/guoxiaobo96/checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
train_rate=0.6
valid_rate=0.2
test_rate=0.2



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch,keep_prob):
    feed_dict = {
        model.x: x_batch,
        model.y: y_batch,
        model.keep_prob:keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 64)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.precision], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(x_train,y_train,x_val,y_val):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'd:/tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.precision)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 3000  # 如果超过3000轮未提升，提前结束训练

    flag = False
    for _ in range(config.num_epochs):
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch,0.8)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.precision], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optimizer, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test(x_test,y_test):
    print("Loading test data...")
    start_time = time.time()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test = np.argmax(y_test, 1)
    y_pred = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred[start_id:end_id] = session.run(model.y_pred, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    #print(metrics.classification_report(y_true=y_test,y_pred=y_pred,target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def final_test(x_test,y_test,x_text):
    print("Loading test data...")
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

#    print('Testing...')
#    loss_test, acc_test = evaluate(session, x_test, y_test)
#   msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
#    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test = np.argmax(y_test, 1)
    y_pred = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred[start_id:end_id] = session.run(model.y_pred, feed_dict=feed_dict)
        count=0
    for i in range(len(x_test)):
        if str(y_pred[i])==str(y_test[i]):
            count+=1
    print(count)
#        with open(file=base_dir+'/'+str(y_pred[i])+'.txt',mode='a',encoding='utf8') as f:
#            f.write(str(x_text[i])+'\n')

if __name__ == '__main__':
#    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
#        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(data_dir, vocab_dir, config.vocab_size,1)
    categories, cat_to_id = read_catagory()
    #words, word_to_id = read_vocab(vocab_dir)
    words,words_to_id,vocab_length,vocab_dim=read_word2vec(vocab_dir)
    config.vocab_size = len(words)
    config.vocab_dim=vocab_dim
    #x_train,y_train,x_val,y_val,x_test,y_test=get_data(data_dir,word_to_id,cat_to_id,config.seq_length)
    x_train,y_train,x_val,y_val,x_test,y_test=get_data_with_vocab(data_dir,words_to_id,config.seq_length)
#    x_train,y_train,_=get_data(train_dir,word_to_id,cat_to_id,config.seq_length,split=False)
#    x_val,y_val,_=get_data(val_dir,word_to_id,cat_to_id,config.seq_length,split=False)
#    x_test,y_test,x_text=get_data(test_dir,word_to_id,cat_to_id,config.seq_length,split=False)
    model = CharLevelCNN(config)
#    if sys.argv[1] == 'train':
    #model =cnnModel.(config,128,2,256,5)
    train(x_train,y_train,x_val,y_val)
#    else:
    test(x_test,y_test)
#    final_test(x_test,y_test,x_text)
