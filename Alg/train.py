#coding:utf-8
import tensorflow as tf
import numpy as np
import scipy.io as sio
import math
import random
from sklearn.cluster import KMeans
# from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Train(object):
    def __init__(self,config,node_walks,attr_walks):
        self.embpath = config.embpath
        self.inputpath = config.inputpath
        self.dimensions = config.dimensions
        self.nodewinsize = config.nodewinsize
        self.attrwinsize = config.attrwinsize
        self.node_walks = node_walks
        self.attr_walks = attr_walks
        self.batch_num = 400
        linkmapfile = open(self.inputpath+"/link_map.txt")
        self.num_nodes = int(linkmapfile.readline().strip().split()[0])
        attrmapfile = open(self.inputpath+"/attr_map.txt")
        self.num_values = int(attrmapfile.readline().strip().split()[0])
        attrcatemapfile = open(self.inputpath+"/attrcate_map.txt")
        self.attrcate = {}
        for line in attrcatemapfile.readlines():
            item = line.strip().split()
            self.attrcate[int(item[0])] =  [int(item[i]) for i in range(1,len(item))]
        self.num_attrs = len(self.attrcate.keys())
        self.num_sampled = 5 # number of negative sample
        self.gamma = 1
        self.lr = 0.001
        self.alpha = 1
        self.theta = 10
        self.bias = 0
        self.config = config
    
    # 产生中心节点和上下文节点的pair
    def gen_pairs(self,walks,winsize):
        
        node_pairs = []
        for walk in walks:
            if(len(walk)>1): # remove isolated nodes
                for j in range(len(walk)):
                    start = max(0,j+1-winsize)
                    end = min(j+1+winsize,len(walk))
                    for i in range(start,end):
                        if(i != j):
                            node_pairs.append([walk[j],walk[i]])
                            
        return node_pairs
    
    # 产生节点、属性类别、正属性值、负属性值四元组
    def gen_quadruples(self):
        
        triplefile = open(self.inputpath+"/attr_train.edgelist")
        quadruples = []
        for line in triplefile.readlines():
            item = line.strip().split()
            triple = [int(item[i]) for i in range(len(item))]
            # negcount = min(self.num_sampled,len(self.attrcate[triple[1]])-1)
            self.attrcate[triple[1]].remove(triple[2])
            # negsamples = random.sample(self.attrcate[triple[1]],negcount)
            negsamples = list(np.random.choice(self.attrcate[triple[1]],self.num_sampled))
            self.attrcate[triple[1]].append(triple[2])
            triple.extend(negsamples)
            quadruples.append(triple)
        
        return quadruples

    # 产生训练batch
    def gen_batch(self,node_pairs,attr_pairs):
        
        node_data_size = len(node_pairs)
        attr_data_size = len(attr_pairs)
        node_indices = np.random.permutation(np.arange(node_data_size))
        attr_indices = np.random.permutation(np.arange(attr_data_size))
        node_batch_size = int(node_data_size/self.batch_num)
        node_start_index = 0
        node_end_index = min(node_start_index+node_batch_size,node_data_size)
        attr_batch_size = int(attr_data_size/self.batch_num)
        attr_start_index = 0
        attr_end_index = min(attr_start_index+attr_batch_size,attr_data_size)
        count = 0
        while(count < self.batch_num):
            node_index = node_indices[node_start_index:node_end_index]
            node_pairs_batch = np.array(node_pairs)[node_index]
            attr_index = attr_indices[attr_start_index:attr_end_index]
            attr_pairs_batch = np.array(attr_pairs)[attr_index]
            yield node_pairs_batch,attr_pairs_batch
            node_start_index = node_end_index
            node_end_index = min(node_start_index+node_batch_size,node_data_size)
            attr_start_index = attr_end_index
            attr_end_index = min(attr_start_index+attr_batch_size,attr_data_size)
            count += 1
    
    # 读取初始化embedding
    def read_initemb(self):
    	
        linkmapfile = open(self.inputpath+"/link_map.txt")
        linkmapfile.readline()
        node2id = {}
        for line in linkmapfile.readlines():
            item = line.strip().split()
            node2id[int(item[0])] = int(item[1])

        initembfile = open(self.inputpath+"/init.emb")
        initembfile.readline()
        initemb = np.array(np.random.normal(0.0, 1.0, (self.num_nodes,self.dimensions)))*1.0/math.sqrt(self.dimensions)
        for line in initembfile.readlines():
            item = line.strip().split()
            initemb[node2id[int(item[0])],:] = np.array([float(item[i]) for i in range(1,len(item))])

        return initemb

    # tensorflow训练过程
    def tf_train_step(self):

        global_step = tf.Variable(0, name='global_step', trainable=False)

        node_pairs = tf.placeholder(tf.int32,shape=[None,2])
        attr_pairs = tf.placeholder(tf.int32,shape=[None,2])
        quadruples = tf.placeholder(tf.int32,shape=[None,3+self.num_sampled])

        # node_embeddings = tf.Variable(tf.assign(self.read_initemb()))
        node_embeddings = tf.Variable(tf.random_uniform([self.num_nodes,self.dimensions],-1.0,1.0))
        value_embeddings = tf.Variable(tf.random_uniform([self.num_values,self.dimensions],-1.0,1.0))
        attr_embeddings = tf.Variable(tf.random_uniform([self.num_attrs+1,self.dimensions],-1.0,1.0))
        
        input_node = tf.reshape(node_pairs[:,0],[-1])
        context_node = tf.reshape(node_pairs[:,1],[-1,1])
        node_embed = tf.nn.embedding_lookup(node_embeddings,input_node)
        node_softmax_biases = tf.zeros([self.num_nodes])
        node_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(node_embeddings,node_softmax_biases,context_node,node_embed,self.num_sampled,self.num_nodes))

        input_attr = tf.reshape(attr_pairs[:,0],[-1])
        context_attr = tf.reshape(attr_pairs[:,1],[-1,1])
        attr_embed = tf.nn.embedding_lookup(value_embeddings,input_attr)
        attr_softmax_biases = tf.zeros([self.num_values])
        attr_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(value_embeddings,attr_softmax_biases,context_attr,attr_embed,self.num_sampled,self.num_values))

        pos_h = tf.nn.l2_normalize(tf.nn.embedding_lookup(node_embeddings,tf.reshape(quadruples[:,0],[-1])),1)
        pos_r = tf.nn.l2_normalize(tf.nn.embedding_lookup(attr_embeddings,tf.reshape(quadruples[:,1],[-1])),1)
        pos_t = tf.nn.l2_normalize(tf.nn.embedding_lookup(value_embeddings,tf.reshape(quadruples[:,2],[-1])),1)
        
        pos = tf.reduce_sum(tf.log_sigmoid(-abs(pos_h + pos_r - pos_t) + self.bias), 1, keepdims = True)
        
        h_ones = tf.multiply(tf.ones_like(quadruples[:,3:]),tf.reshape(quadruples[:,0],[-1,1]))
        neg_h = tf.nn.l2_normalize(tf.nn.embedding_lookup(node_embeddings,h_ones),2)
        r_ones = tf.multiply(tf.ones_like(quadruples[:,3:]),tf.reshape(quadruples[:,1],[-1,1]))
        neg_r = tf.nn.l2_normalize(tf.nn.embedding_lookup(attr_embeddings,r_ones),2)
        neg_t = tf.nn.l2_normalize(tf.nn.embedding_lookup(value_embeddings,quadruples[:,3:]),2)
        
        neg = tf.reduce_mean(tf.reduce_sum(tf.log_sigmoid(abs(neg_h + neg_r - neg_t) + self.bias), 2, keepdims = True),1)
        
        trans_loss = tf.reduce_sum(pos + neg)

        loss = node_loss + self.alpha*attr_loss + self.theta*trans_loss
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss,global_step=global_step)

        node_pairs_all = self.gen_pairs(self.node_walks,self.nodewinsize)
        attr_pairs_all = self.gen_pairs(self.attr_walks,self.attrwinsize)
        batches = self.gen_batch(node_pairs_all,attr_pairs_all)
        initemb = self.read_initemb()

        init = tf.global_variables_initializer()
        print("tensorflow train step...")
        with tf.Session() as sess:
            sess.run(init)
            sess.run(tf.assign(node_embeddings,initemb))
            for batch_id,batch in enumerate(batches):
                    node_pairs_batch,attr_pairs_batch= batch
                    quadruples_batch = self.gen_quadruples()
                    feed_dict = {node_pairs:node_pairs_batch,attr_pairs:attr_pairs_batch,quadruples:quadruples_batch}
                    _,batch_loss,nodeembs,attrembs,valueembs= sess.run([optimizer,loss,node_embeddings,attr_embeddings,value_embeddings],feed_dict=feed_dict)
                    print("%3d/%d:  %f"%(batch_id+1,self.batch_num,batch_loss))

        sio.savemat(self.embpath,{'node':nodeembs,'attr':attrembs,'value':valueembs})

class Eval(object):
    def __init__(self,config):
        self.embpath = config.embpath
        self.inputpath = config.inputpath
        linkmapfile = open(self.inputpath+"/link_map.txt")
        self.num_nodes = int(linkmapfile.readline().strip().split()[0])
        attrcatemapfile = open(self.inputpath+"/attrcate_map.txt")
        self.attrcate = {}
        for line in attrcatemapfile.readlines():
            item = line.strip().split()
            self.attrcate[int(item[0])] =  [int(item[i]) for i in range(1,len(item))]

    # 节点分类函数
    def node_classify(self):
        print("node classify...")
        data = sio.loadmat(self.embpath+".mat")
        labelfile = open(self.inputpath+"/group.txt")
        linkmapfile = open(self.inputpath+"/link_map.txt")
        linkmapfile.readline()
        node2id = {}
        for line in linkmapfile.readlines():
            item = line.strip().split()
            node2id[int(item[0])] = int(item[1])
        sorted_node2id = sorted(node2id.items(),key=lambda k:k[1])
        d = [node[0] for node in sorted_node2id]
        label = []
        for line in labelfile.readlines():
            item = line.strip().split()
            label.append(int(item[0]))

        label = np.array(label)[d]
        # label = data['label'].ravel()
        emb = data['node']
        
        micro_list = []
        # macro_list = []
        for test_size in [0.85,0.75,0.65,0.55,0.45,0.35,0.25]:
            x_train,x_test,y_train,y_test = train_test_split(emb, label, random_state=1, test_size=test_size)
            clf = svm.SVC(C=100,gamma="scale",kernel='rbf')
            clf.fit(x_train,y_train)
            y_test_hat = clf.predict(x_test)
            micro_list.append(str(np.round(metrics.f1_score(y_test,y_test_hat,average='micro')*10000)/100))
            # macro_list.append(str(np.round(metrics.f1_score(y_test,y_test_hat,average='macro')*10000)/100))
        print("node classfication...")
        print("F1-score(micro): %s"%(" ".join(micro_list)))
        # print("F1-score(macro): %s"%(" ".join(macro_list)))

    # 网络可视化函数
    def network_visualization(self,dataname):
        print("network visualization...")
        data = sio.loadmat(self.embpath+".mat")
        labelfile = open(self.inputpath+"/group.txt")
        linkmapfile = open(self.inputpath+"/link_map.txt")
        linkmapfile.readline()
        node2id = {}
        for line in linkmapfile.readlines():
            item = line.strip().split()
            node2id[int(item[0])] = int(item[1])
        sorted_node2id = sorted(node2id.items(),key=lambda k:k[1])
        d = [node[0] for node in sorted_node2id]
        label = []
        for line in labelfile.readlines():
            item = line.strip().split()
            label.append(int(item[0]))

        label = np.array(label)[d]
        # label = data['label'].ravel()
        emb = data['node']

        emb_tsne = TSNE().fit_transform(emb)

        plt.scatter(emb_tsne[:,0],emb_tsne[:,1],10*label,10*label)
        plt.savefig(dataname+".png")