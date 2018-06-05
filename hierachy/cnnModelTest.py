import tensorflow as tf
import pickle


class TCNNConfig(object):
    """CNN配置参数"""
    vocab_dim=400
    embedding_size = 128  # 词向量维度
    seq_length = 500  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    filter_size = [2,3,4,5]  # 卷积核尺寸
    vocab_size = 5000  # 词汇表大小

    hidden_dim = 1024  # 全连接层神经元

    learning_rate = 1e-5 # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10000  # 总迭代轮次
    kernel_size = 5
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    l2_lambda=0.1


class CharLevelCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.x = tf.placeholder(tf.int32, [None,self.config.vocab_dim ,self.config.seq_length], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.l2_loss=0
        self.l2_lambda=0.2

        self.char_level_cnn()

    def char_level_cnn(self):
        """CNN模型"""
        # 词向量映射

        with tf.name_scope("cnn"):
            # CNN layer
            conv_1 = tf.layers.conv1d(embedding_inputs, 64, 3, name='conv1',trainable=True,padding='same')
            conv_1=tf.nn.relu(conv_1)
            # global max pooling layer
            max_pool_1=tf.layers.max_pooling1d(conv_1,2,strides=1,padding='same')
            conv_2=tf.layers.conv1d(max_pool_1,64, 1, name='conv2')
            conv_3=tf.layers.conv1d(conv_2,32, 3, name='conv3')
            max_pool_3=tf.layers.max_pooling1d(conv_3,1,strides=1,padding='same')
            gmp=tf.reduce_max(max_pool_3, reduction_indices=[1], name='gmp')
            

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            W=tf.get_variable("W",shape=[self.config.hidden_dim,self.config.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits=tf.nn.xw_plus_b(fc,W,b)
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
            self.loss=tf.reduce_mean(cross_entropy)+self.l2_loss*self.l2_lambda
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.y, 1), self.y_pred)
            self.precision = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

class TextCnn(object):
    def __init__(self,config):  
        self.config=config
        self.batch_size = config.batch_size
        # 词典的大小
        self.vocab_size = config.vocab_size
        self.num_classes = config.num_classes
        # length of word embedding
        self.embedding_size = config.embedding_size
        # seting filter sizes, type of list
        self.filter_sizes = config.filter_size
        # max length of sentence
        self.sentence_length = config.seq_length
        # number of filters
        self.num_filters = config.num_filters
        self.learning_rate=config.learning_rate
        self.num_classes=config.num_classes
        self.build_graph()
    
    def __add_placeholders(self):
        self.x=tf.placeholder('int32',[None,self.sentence_length],name='x')
        self.y=tf.placeholder('int32',[None,self.num_classes],name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    def __inference(self):
        with tf.variable_scope('embedding_layer'):
            self.W=tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,-1.0,),name='embedding_weights',dtype='float32')
            self.embedding_chars=tf.nn.embedding_lookup(self.W,self.x)
            self.embedding_chars_expend=tf.expand_dims(self.embedding_chars,-1)

        with tf.variable_scope('convolution_pooling_layer'):
            pooled_outputs=[]
            for _,filter_size in enumerate(self.filter_sizes):
                conv_1 = tf.layers.conv1d(self.embedding_chars,256, filter_size,trainable=True,padding='valid')
                conv_1=tf.nn.relu(conv_1)
                pooled_1=tf.layers.max_pooling1d(conv_1,3,strides=1,padding='same')
                conv_2 = tf.layers.conv1d(pooled_1,256, filter_size-1,trainable=True,padding='valid')
                conv_2=tf.nn.relu(conv_2)
                conv_3 = tf.layers.conv1d(conv_2,256, filter_size-1,trainable=True,padding='valid')
                conv_3=tf.nn.relu(conv_3)
                pooled_3=tf.layers.max_pooling1d(conv_3,2,strides=1,padding='same')
                pooled_3=tf.reduce_max(pooled_3, reduction_indices=[1])
                pooled_outputs.append(pooled_3)
            self.feature_length=self.num_filters*len(self.filter_sizes)
            self.h_pool=tf.concat(pooled_outputs,1)
            self.h_pool_flat=tf.reshape(self.h_pool,[-1,self.feature_length])
        
        with tf.variable_scope('drop_out_layer'):
            self.features=tf.nn.dropout(self.h_pool_flat,self.keep_prob)
            
        with tf.variable_scope('fully_connected_layer'):
            self.features=tf.nn.relu(self.features)
            self.y_out=tf.layers.dense(self.features,self.num_classes)
            self.y_prob=tf.nn.softmax(self.y_out,1)
    
    def __add_metric(self):
        self.y_pred=tf.argmax(self.y_prob,1)
        correct_pred=tf.equal(self.y_pred,tf.argmax(self.y,1))
        self.precision=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        self.recall, self.recall_op = tf.metrics.recall(tf.argmax(self.y,1), self.y_pred)
        tf.summary.scalar('precision', self.precision)
        tf.summary.scalar('recall', self.recall)

    def __add_loss(self):
        loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_out, labels=self.y)
        self.loss=tf.reduce_mean(loss)
        tf.summary.scalar('loss',self.loss)
    

    def __train(self):
        self.global_step=tf.Variable(0,trainable=False)
        self.optimizer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        #extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(extra_update_ops):
            #self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    
    def build_graph(self):
        self.__add_placeholders()
        self.__inference()
        self.__add_metric()
        self.__add_loss()
        self.__train()

class TestModel(object):
    def __init__(self,config,rnn_size,layer_size,attention_size,grad_clip):  
        self.config=config
        self.batch_size = config.batch_size
        # 词典的大小
        self.vocab_size = config.vocab_size
        self.num_classes = config.num_classes
        # length of word embedding
        self.embedding_size = config.embedding_size
        # seting filter sizes, type of list
        self.rnn_size=rnn_size
        self.layer_size=layer_size
        self.filter_sizes = config.filter_size
        # max length of sentence
        self.sequence_length = config.seq_length
        # number of filters
        self.num_filters = config.num_filters
        self.learning_rate=config.learning_rate
        self.num_classes=config.num_classes
        self.attention_size=attention_size
        self.grad_clip=grad_clip
        self.l2_loss=0
        self.l2_lambda=config.l2_lambda
        self.build_graph()
    
    def __add_placeholders(self):
        self.x=tf.placeholder('int32',[None,self.sequence_length],name='x')
        self.y=tf.placeholder('int32',[None,self.num_classes],name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    def __inference(self):
        with tf.variable_scope('embedding_layer'):
            embedding=tf.Variable(tf.truncated_normal([self.vocab_size,self.embedding_size],stddev=0.1),name='embedding')
            embedding_inputs=tf.nn.embedding_lookup(embedding,self.x)
            embedding_inputs=tf.transpose(embedding_inputs,[1,0,2])
            embedding_inputs=tf.reshape(embedding_inputs,[-1,self.rnn_size])
            embedding_inputs=tf.split(embedding_inputs,self.sequence_length,0)


        with tf.name_scope('fw_lstm'),tf.variable_scope('fw_lstm'):
            print(tf.get_variable_scope().name)
            lstm_fw_cell_list=[tf.nn.rnn_cell.LSTMCell(self.rnn_size) for _ in range(self.layer_size)]
            lstm_fw_cell_m=tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(lstm_fw_cell_list),output_keep_prob=self.keep_prob)
        
        with tf.name_scope('bw_lstm'),tf.variable_scope('bw_lstm'):
            print(tf.get_variable_scope().name)
            lstm_bw_cell_list=[tf.nn.rnn_cell.LSTMCell(self.rnn_size) for _ in range(self.layer_size)]
            lstm_bw_cell_m=tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(lstm_bw_cell_list),output_keep_prob=self.keep_prob)
            
        with tf.name_scope('bi_lstm'),tf.variable_scope('bi_lstm'):
            self.lstm_out,_,_=tf.nn.static_bidirectional_rnn(lstm_bw_cell_m,lstm_fw_cell_m,embedding_inputs,dtype=tf.float32)

        with tf.name_scope('attention'),tf.variable_scope('attention'):
            attention_w=tf.Variable(tf.truncated_normal([2*self.rnn_size,self.attention_size],stddev=0.1),name='attention_w')
            attention_b=tf.Variable(tf.constant(0.1,shape=[self.attention_size]),name='attention_b')
            u_list=[]
            for t in range(self.sequence_length):
                u_t=tf.tanh(tf.matmul(self.lstm_out[t],attention_w)+attention_b)
                u_list.append(u_t)
            u_w=tf.Variable(tf.truncated_normal([self.attention_size,1],stddev=0.1),name='attention_uw')
            attention_z=[]
            for t in range(self.sequence_length):
                z_t=tf.matmul(u_list[t],u_w)
                attention_z.append(z_t)
            attention_zoncat=tf.concat(attention_z,axis=1)
            attention_alpha=tf.nn.softmax(attention_zoncat)
            alpha_trans=tf.reshape(tf.transpose(attention_alpha,[1,0]),[self.sequence_length,-1,1])
            self.attention_output=tf.reduce_sum(self.lstm_out*alpha_trans,0)
            print(self.attention_output.get_shape())

        with tf.variable_scope('output'):
            fc_w=tf.Variable(tf.truncated_normal([2*self.rnn_size,self.num_classes],stddev=0.1),name='fc_w')
            fc_b=tf.Variable(tf.zeros(self.num_classes),name='fc_b')
            self.y_out=tf.nn.xw_plus_b(self.attention_output,fc_w,fc_b)
            self.y_prob=tf.nn.softmax(self.y_out,1)
    
    def __add_metric(self):
        self.y_pred=tf.argmax(self.y_prob,1)
        correct_pred=tf.equal(self.y_pred,tf.argmax(self.y,1))
        self.precision=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        self.recall, self.recall_op = tf.metrics.recall(tf.argmax(self.y,1), self.y_pred)
        tf.summary.scalar('precision', self.precision)
        tf.summary.scalar('recall', self.recall)

    def __add_grads(self):
        loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_out, labels=self.y)
        self.loss=tf.reduce_mean(loss)
        tf.summary.scalar('loss',self.loss)
        self.t_vars=tf.trainable_variables()
        self.grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,self.t_vars),self.grad_clip)
    

    def __train(self):
        self.optimizer=tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.grads,self.t_vars))
        

    
    def build_graph(self):
        self.__add_placeholders()
        self.__inference()
        self.__add_metric()
        self.__add_grads()
        self.__train()

class TestCnnConv2(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.l2_loss=0
        self.l2_lambda=config.l2_lambda
        self.num_classes=self.config.num_classes
        self.char_level_cnn()

    def char_level_cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            self.W=tf.Variable(tf.random_uniform([self.config.vocab_size,self.config.embedding_size],-1.0,-1.0,),name='embedding_weights',dtype='float32')
            self.embedding_chars=tf.nn.embedding_lookup(self.W,self.x)
            self.embedding_chars_expend=tf.expand_dims(self.embedding_chars,-1)

        with tf.variable_scope('convolution_pooling_layer'):
            pooled_outputs=[]
            for i,filter_size in enumerate(self.config.filter_size):
                filter_shape=[filter_size,self.config.embedding_size,1,1]
                W=tf.get_variable('W'+str(i),shape=filter_shape,initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('b'+str(i), shape=[self.config.num_filters],initializer=tf.zeros_initializer())
                conv=tf.nn.conv2d(self.embedding_chars_expend,W,strides=[1,1,1,1],padding='VALID',name='conv'+str(i))
                h=tf.nn.relu(tf.add(conv,b))
                pooled=tf.nn.max_pool(h,ksize=[1,self.config.seq_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')
                pooled_outputs.append(pooled)
            self.feature_length=self.config.num_filters*len(self.config.filter_size)
            self.h_pool=tf.concat(pooled_outputs,3)
            self.h_pool_flat=tf.reshape(self.h_pool,[-1,self.feature_length])

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(self.h_pool_flat, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc=tf.nn.relu(fc)

            # 分类器
            self.logits=tf.layers.dense(fc,self.num_classes,name="fc2")
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
            self.loss = tf.reduce_mean(cross_entropy)+self.l2_loss*self.l2_lambda
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.y, 1), self.y_pred)
            self.precision = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

class HierachyCnn(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.l2_loss=0
        self.l2_lambda=0.1

        self.char_level_cnn()

    def char_level_cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_size])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv_1 = tf.layers.conv1d(embedding_inputs, 64, 3, name='conv1',trainable=True,padding='same')
            conv_1=tf.nn.relu(conv_1)
            # global max pooling layer
            max_pool_1=tf.layers.max_pooling1d(conv_1,2,strides=1,padding='same')
            conv_2=tf.layers.conv1d(max_pool_1,64, 1, name='conv2')
            conv_3=tf.layers.conv1d(conv_2,32, 3, name='conv3')
            max_pool_3=tf.layers.max_pooling1d(conv_3,1,strides=1,padding='same')
            gmp=tf.reduce_max(max_pool_3, reduction_indices=[1], name='gmp')
            

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            W=tf.get_variable("W",shape=[self.config.hidden_dim,23],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[23]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits=tf.nn.xw_plus_b(fc,W,b)
            temp=[]
            temp.append(tf.slice(self.y,[0,0],[-1,1]))
            temp.append(tf.slice(self.y,[0,1],[-1,9]))
            temp.append(tf.slice(self.y,[0,10],[-1,9]))
            temp.append(tf.slice(self.y,[0,20],[-1,1]))
            temp.append(tf.slice(self.y,[0,21],[-1,1]))
            for i in range(4):
                temp[i]=tf.reduce_mean(temp[i],1,keepdims=True)
            self.logits=tf.concat([temp[0],temp[1],temp[2],temp[3],temp[4]],axis=1)
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
            self.loss=tf.reduce_mean(cross_entropy)+self.l2_loss*self.l2_lambda
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.y, 1), self.y_pred)
            self.precision = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

class TestHierachyCnn(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.l2_loss=0
        self.l2_lambda=0.1

        self.char_level_cnn()

    def char_level_cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_size])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.x)
            temp=[]
            temp.append(tf.slice(self.y,[0,0],[-1,1]))
            temp.append(tf.slice(self.y,[0,1],[-1,9]))
            temp.append(tf.slice(self.y,[0,10],[-1,9]))
            temp.append(tf.slice(self.y,[0,20],[-1,1]))
            temp.append(tf.slice(self.y,[0,21],[-1,1]))
            for i in range(4):
                temp[i]=tf.reduce_mean(temp[i],1,keepdims=True)
            self.y_1=tf.concat([temp[0],temp[1],temp[2],temp[3],temp[4]],axis=1)

        with tf.name_scope("cnn"):
            # CNN layer
            conv_1 = tf.layers.conv1d(embedding_inputs, 64, 3, name='conv1',trainable=True,padding='same')
            conv_1=tf.nn.relu(conv_1)
            # global max pooling layer
            max_pool_1=tf.layers.max_pooling1d(conv_1,2,strides=1,padding='same')
            conv_2=tf.layers.conv1d(max_pool_1,64, 1, name='conv2')
            conv_3=tf.layers.conv1d(conv_2,32, 3, name='conv3')
            max_pool_3=tf.layers.max_pooling1d(conv_3,1,strides=1,padding='same')
            gmp=tf.reduce_max(max_pool_3, reduction_indices=[1], name='gmp')
            

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            W=tf.get_variable("W",shape=[self.config.hidden_dim,23],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[23]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits=tf.nn.xw_plus_b(fc,W,b)
            temp=[]
            temp.append(tf.slice(self.logits,[0,0],[-1,1]))
            temp.append(tf.slice(self.logits,[0,1],[-1,9]))
            temp.append(tf.slice(self.logits,[0,10],[-1,9]))
            temp.append(tf.slice(self.logits,[0,20],[-1,1]))
            temp.append(tf.slice(self.logits,[0,21],[-1,1]))
            for i in range(4):
                temp[i]=tf.reduce_mean(temp[i],1,keepdims=True)
            self.logits_1=tf.concat([temp[0],temp[1],temp[2],temp[3],temp[4]],axis=1)
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits_1), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
            self.loss=tf.reduce_mean(cross_entropy)+self.l2_loss*self.l2_lambda
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.y_1, 1), self.y_pred)
            self.precision = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def main():
    config=TCNNConfig()
    model=TestModel(config,128,2,256,5)

if __name__=='__main__':
    main()