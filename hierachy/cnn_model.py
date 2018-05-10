import tensorflow as tf
import pickle


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_size = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    filter_size = [3,4,5]  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class CharLevelCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.char_level_cnn()

    def char_level_cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

class TextCnn(object):
    def __init__(self,config):  
        self.batch_size = config['batch_size']
        # 词典的大小
        self.vocab_size = config['vocab_size']
        self.num_classes = config['num_classes']
        self.keep_prob = config['keep_prob']
        # length of word embedding
        self.embedding_size = config['embedding_size']
        # seting filter sizes, type of list
        self.filter_sizes = config['filter_sizes']
        # max length of sentence
        self.sentence_length = config['seq_length']
        # number of filters
        self.num_filters = config['num_filters']
        self.learning_rate=config['learning_rate']
    
    def __add_placeholders(self):
        self.x=tf.placeholder('int32',[None,self.sentence_length])
        self.y=tf.placeholder('int32',[None,])
    
    def __inference(self):
        with tf.variable_scope('embedding layer'):
            with open('/embedding_matrix,pkl','rb') as embedding_file:
                embedding_weights=pickle.load(embedding_file)
            self.W=tf.Variable(embedding_weights,trainable=True,name='embedding_weights',dtype='float32')
            self.embedding_chars=tf.nn.embedding_lookup(self.W,self.x)
            self.embedding_chars_expend=tf.expand_dims(self.embedding_chars,-1)

        with tf.variable_scope('convolution_pooling_layer'):
            pooled_outputs=[]
            for i,filter_size in enumerate(self.filter_sizes):
                filter_shape=[filter_size,self.sentence_length-filter_size+1,1,1]
                W=tf.get_variable('W'+str(i),shape=filter_shape,initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('b'+str(i), shape=[self.num_filters],initializer=tf.zeros_initializer())
                conv=tf.nn.conv2d(self.embedding_chars_expend,W,strides=[1,1,1,1],padding='VALID',name='conv'+str(i))
                h=tf.nn.relu(tf.add(conv,b))
                pooled=tf.nn.max_pool(h,ksize=[1,self.sentence_length-self.filter_sizes+1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')
                pooled_outputs.append(pooled)
            self.feature_length=self.num_filters*len(self.filter_sizes)
            self.h_pool=tf.concat(pooled_outputs,3)
            self.h_pool_flat=tf.reshape(self.h_pool,-1,self.feature_length)
        
        with tf.variable_scope('drop_out_layer'):
            self.features=tf.nn.dropout(self.h_pool_flat,self.keep_prob)
            
        with tf.variable_scope('fully_connected_layer'):
            w=tf.get_variable('W',shape=[self.feature_length,self.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable('b',shape=[self.num_classes],initializer=tf.constant_initializer(0.1))
            self.y_out=tf.matmul(self.features,w)+b
            self.y_prob=tf.nn.softmax(self.y_out)
    
    def __add_loss(self):
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.y_pred)
        self.loss=tf.reduce_mean(loss)
        tf.summary.scalar('loss',self.loss)
    
    def __add_metric(self):
        self.y_pred=self.y_prob[:,1]>0.5
        self.precision, self.precision_op = tf.metrics.precision(self.y, self.y_pred)
        self.recall, self.recall_op = tf.metrics.recall(self.y, self.y_pred)
        tf.summary.scalar('precision', self.precision)
        tf.summary.scalar('recall', self.recall)

    def __train(self):
        self.global_step=tf.Variable(0,trainable=False)
        optimizer=tf.train.AdamOptimizer(self.learning_rate)
        extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
    
    def build_graph(self):
        self.__add_placeholders()
        self.__inference()
        self.__add_loss()
        self.__add_metric()
        self.__train()