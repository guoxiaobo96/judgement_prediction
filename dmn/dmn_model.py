import sys

import numpy as np
import tensorflow as tf
from attenttion_gru_cell import AttentionGRUCell
import data_input


class Config():
    batch_size = 64
    embedding_size = 80
    hidden_size = 80

    max_epoch = 256
    early_stopping = 20

    drop_out = 0.9
    lr = 0.001
    l2 = 0.001

    cap_grads = False
    max_grad_val = 10
    noisy_grads = False

    word2vec_init = False
    embedding_init = np.sqrt(3)

    # NOTE not currently used hence non-sensical anneal_threshold
    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 4

    max_allowed_inputs = 130
    num_train = 9000

    floatX = np.float32

    train_mode = True


def _add_gradient_noise(tensor, stddev=1e-3):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `tensor` should be a gradient.
    The output will be `tensor` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.variable_scope('gradient_noise'):
        gradient_noise = tf.random_normal(tf.shape(tensor), stddev=stddev)
        return tf.add(tensor, gradient_noise)

# from https://github.com/domluna/memn2n


def _position_encoding(sentence_size, embedding_size):
    """We could have used RNN for parsing sentence but that tends to overfit.
    The simpler choice would be to take sum of embedding but we loose loose positional information.
    Position encoding is described in section 4.1 in "End to End Memory Networks" in more detail (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    sentence_len = sentence_size+1
    embedding_len = embedding_size+1
    for i in range(1, embedding_len):
        for j in range(1, sentence_len):
            encoding[i-1, j-1] = (i - (embedding_len-1)/2) * \
                (j - (sentence_len-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


class DmnModel():
    def __init__(self, config):
        self.config = config
        self.variables_to_save = {}
        self.load_data()
        self.add_placeholders()

        self.embeddings = tf.Variable(
            self.word_embedding.astype(np.float32), name="embedding")

        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.summary.merge_all()

    def load_data(self):
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test, self.question, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size, self.question_num = data_input.load_data(
            self.config)
        self.encoding = _position_encoding(
            self.max_sen_len, self.config.embed_size)
        self.train = zip(self.question, self.x_train,
                         self.max_q_len, self.max_sentences, self.y_train)
        self.valid = zip(self.question, self.x_valid,
                         self.max_q_len, self.max_sentences, self.y_valid)
        self.test = zip(self.question, self.x_test,
                        self.max_q_len, self.max_sentences, self.y_test)

    def add_placeholders(self):
        self.question_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size, self.max_q_len, self.question_num))
        self.input_placeholder = tf.placeholder(tf.int32, shape=(
            self.config.batch_size, self.max_sentences, self.max_sen_len))

        self.question_len_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(
            tf.int32, shape=(self.config.batch_size,))

        self.answer_placeholder = tf.placeholder(
            tf.int64, shape=(self.config.batch_size,))

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_predictions(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred

    def add_loss_op(self, output):
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=self.answer_placeholder))
        for variable in tf.trainable_variables():
            if not 'bias' in variable.name.lower():
                loss += self.config.l2*tf.nn.l2_loss(variable)
        tf.summary.scalar('loss', loss)

    def add_training_op(self, loss):
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var)
                   for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op

    def get_question_representation(self):
        questions = tf.nn.embedding_lookup(
            self.embeddings, self.question_placeholder)
        gru_cell = tf.contrib.rnn.gru_cell(self.config.hidden_size)
        _, question_vec = tf.nn.dynamic_rnn(
            gru_cell, questions, dtype=np.float32, sequence_length=self.question_len_placeholder)
        return question_vec

    def get_input_representation(self):
        inputs = tf.nn.embedding_lookup(
            self.embeddings, self.input_placeholder)

        inputs = tf.reduce_sum(inputs*self.encoding, 2)
        forward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)

        out_puts, _ = tf.nn.bidirectional_dynamic_rnn(
            forward_gru_cell, backward_gru_cell, inputs, dtype=np.float32, sequence_length=self.input_len_placeholder)

        input_vector = tf.reduce_sum(tf.stack(out_puts), axis=0)
        input_vector = tf.nn.dropout(input_vector, self.dropout_placeholder)

        return input_vector

    def get_attention(self, question_vec, prev_memory, input_vector, reuse):
        with tf.variable_scope("attention", reuse=reuse):
            features = [input_vector*question_vec, input_vector*prev_memory,
                        tf.abs(input_vector-question_vec), tf.abs(input_vector-prev_memory)]
            features_vec = tf.concat(features, 1)
            attention = tf.contrib.layers.fully_connected(
                features_vec, self.config.embedding_size, activation_fn=tf.nn.tanh, reuse=reuse, scope="fc1")
            attention = tf.contrib.layers.fully_connected(
                attention, 1, activation_fn=None, reuse=reuse, scope="fc2")
        return attention

    def generate_episode(self, memory, question_vector, input_vectors, hop_index):
        attentions = [tf.squeeze(self.get_attention(question_vector, memory, fv, bool(
            hop_index) or bool(i)), axis=1)for i, fv in enumerate(tf.unstack(input_vectors, axis=1))]
        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)
        reuse = True if hop_index > 0 else False

        gru_inputs = tf.concat([input_vectors, attentions], 2)
        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(
                self.config.hidden_size), gru_inputs, dtype=np.float32, sequence_length=self.input_len_placeholder)
        return episode

    def add_answer_module(self, rnn_output, question_vector):
        rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)
        output = tf.layers.dense(
            tf.concat([rnn_output, question_vector], 1), self.vocab_size, activation=None)

        return output

    def inference(self):
        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get question representation')
            question_vector = self.get_question_representation()

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            input_vectors = self.get_input_representation()

        self.attentions = []

        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')
            prev_memory = question_vector
            for i in range(self.config.num_hops):
                print('==> generating episode', i)
                episode = self.generate_episode(
                    prev_memory, question_vector, input_vectors, i)

                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat(
                        [prev_memory, episode, question_vector], 1), self.config.hidden_size, activation=tf.nn.relu)
            out_put = prev_memory
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            out_put = self.add_answer_module(out_put, question_vector)

        return out_put

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2):
        config = self.config
        drop_out = config.drop_out
        if train_op is None:
            train_op = tf.no_op()
            drop_out = 1
        total_steps = len(data[0])//config.batch_size
        total_loss = []
        accuracy = 0
        partition = np.random.permutation(len(data[0]))
        question_place, input_place, question_len, input_len, answer_place = data
        question_place, input_place, question_len, input_len, answer_place = question_place[
            partition], input_place[partition], question_len[partition], input_len[partition], answer_place[partition]

        for step in range(total_steps):
            index = range(step*config.batch_size, (step+1)*config.batch_size)
            feed = {self.question_placeholder: question_place[index],
                    self.input_placeholder: input_place[index],
                    self.question_len_placeholder: question_len[index],
                    self.input_len_placeholder: input_len[index],
                    self.answer_placeholder: answer_place[index],
                    self.dropout_placeholder: drop_out}
            loss, pred, summary, _ = session.run(
                [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

            answers = answer_place[step *
                                   config.batch_size:(step+1)*config.batch_size]
            accuracy += np.sum(pred == answers)/float(len(answers))

            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        return np.mean(total_loss), accuracy/float(total_steps)
