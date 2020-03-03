#coding:utf-8
import networkx as nx
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# 输入文件
# doublelink.edgelist 边文件，格式为head_id tail_id,以空格间隔
# attr_info.txt 节点属性文件，格式为node_id 属性类别1_属性值 属性类别2_属性值 属性类别3_属性值 ...
# group.txt 节点label文件，每一行表示一个id的label
# init.emb 节点embedding的初始化文件，以deepwalk pre-train产生的embedding作为初始化embedding
class Graph(object):
    def __init__(self,config):
        
        self.inputpath = config.inputpath
        self.nodewalklen = config.nodewalklen
        self.attrwalklen = config.attrwalklen
        self.walknum = config.walknum
        self.p = config.p
        self.q = config.q
    
    # 预处理函数，将原始node_id映射为从0开始，选取一列作为label，将属性（属性类别_属性值）建立id映射表
    def pre_processing(self):
    	
        links = []
        nodes = []
        node2id = {}
        
        print("read edgefile...")
        link_info = open(self.inputpath+"/doublelink.edgelist",'r')
        nodes_num = 0
        for line in link_info.readlines():
            link = line.strip().split()
            for i in range(2):
                if int(link[i]) not in nodes:
                    nodes.append(int(link[i]))
                    node2id[int(link[i])] = nodes_num
                    nodes_num += 1
            links.append([node2id[int(link[0])],node2id[int(link[1])]])

        print("read attrfile...")
        attr_info = open(self.inputpath+"/attr_info.txt",'r')
        attrs_num = 0
        attr_map = {}
        attr_triples = {}
        attr2id = {}
        attrs = []
        for line in attr_info.readlines():
            attr = line.strip().split()
            for i in range(1,len(attr)):
                attr_value = attr[i]
                attr_value_split = attr_value.split('_')
                if(attr_value_split[0] != '0'):
                    if attr_value not in attrs:
                        attrs.append(attr_value)
                        attr2id[attr_value] = attrs_num
                        attrs_num += 1
                    # links.append([node2id[int(attr[0])],node2id[attr_value]])
                    
                    if(attr_value_split[0] not in attr_map):
                        attr_map[attr_value_split[0]] = []
                        attr_triples[attr_value_split[0]] = []
                    else:
                        attr_triples[attr_value_split[0]].append([node2id[int(attr[0])],int(attr_value_split[0]),attr2id[attr_value]])
                        if(attr2id[attr_value] not in attr_map[attr_value_split[0]]):
                            attr_map[attr_value_split[0]].append(attr2id[attr_value])

        print("write files...")
        # attr_triples_train_0,attr_triples_test_0 = train_test_split(attr_triples['0'], random_state=1, test_size=self.test_size)
        # attr_triples_train_1,attr_triples_test_1 = train_test_split(attr_triples['1'], random_state=1, test_size=self.test_size)
        # attr_triples_train = attr_triples_train_0 + attr_triples['1']+ attr_triples['5'] + attr_triples['2'] + attr_triples['3'] + attr_triples['4'] + attr_triples['6']
        # attr_triples_test = attr_triples_test_0 #+ attr_triples_test_1
        attr_triples_train = attr_triples['1'] + attr_triples['5'] + attr_triples['2'] + attr_triples['3'] + attr_triples['4'] + attr_triples['6']

        trainedgefile = open(self.inputpath+"/link_train.edgelist",'w')
        for link in links:
            trainedgefile.write("%d\t%d\n"%(link[0],link[1]))
        
        trainattrfile = open(self.inputpath+"/attr_train.edgelist",'w')
        for link in attr_triples_train:
            trainattrfile.write("%d\t%d\t%d\n"%(link[0],link[1],link[2]))

        # testattrfile = open(self.inputpath+"/attr_test.edgelist",'w')
        # for link in attr_triples_test:
        #     testattrfile.write("%d\t%d\t%d\n"%(link[0],link[1],link[2]))

        node_attrs = {}
        for link in attr_triples_train:
            if(link[0] not in node_attrs):
                node_attrs[link[0]] = []
            else:
                node_attrs[link[0]].append(link[2])
        attr_links = []
        for node in node_attrs:
        	for i in range(len(node_attrs[node])):
        		for j in range(i+1,len(node_attrs[node])):
        			attr_links.append(tuple(sorted([node_attrs[node][i],node_attrs[node][j]])))
        attr_links = dict(Counter(attr_links))
        
        attrlinkfile = open(self.inputpath+"/attr_links_train.edgelist","w")
        for link in attr_links:
        	attrlinkfile.write("%d\t%d\t%d\n"%(link[0],link[1],attr_links[link]))

        nodemapfile = open(self.inputpath+"/link_map.txt","w")
        nodemapfile.write("%d\n"%(nodes_num))
        for node in node2id:
            nodemapfile.write("%s\t%d\n"%(node,node2id[node]))

        attrmapfile = open(self.inputpath+"/attr_map.txt","w")
        attrmapfile.write("%d\n"%(attrs_num))
        for attr in attr2id:
            attrmapfile.write("%s\t%d\n"%(attr,attr2id[attr]))

        attrcatemapfile = open(self.inputpath+"/attrcate_map.txt","w")
        for attr in attr_map:
            attrcatemapfile.write("%s\t"%attr)
            for value in attr_map[attr]:
                attrcatemapfile.write("%d\t"%value)
            attrcatemapfile.write("\n")

    # 读取node网络图和attr网络图
    def read_graph(self):
        
        print("read linkgraphfile...")
        graphpath = self.inputpath+"/link_train.edgelist"
        self.G = nx.read_edgelist(graphpath,nodetype=int,create_using=nx.DiGraph())
        for edge in self.G.edges():
            self.G[edge[0]][edge[1]]['weight'] = 1
        self.G = self.G.to_undirected()
        
        mapfile = open(self.inputpath+"/link_map.txt","r")
        item = mapfile.readline().strip().split()
        self.num_nodes = int(item[0])

        print("read attrgraphfile...")
        graphpath = self.inputpath+"/attr_links_train.edgelist"
        self.G1 = nx.read_edgelist(graphpath, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
        self.G1 = self.G1.to_undirected()
        
        mapfile = open(self.inputpath+"/attr_map.txt","r")
        item = mapfile.readline().strip().split()
        self.num_attrs = int(item[0])

    # node节点随机游走
    def gen_node_walk(self,startnode):
        
        G = self.G
        walklen = self.nodewalklen
        
        walk = [startnode]
        
        while len(walk) < walklen:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if(len(cur_nbrs)>0):
                walk.append(random.choice(cur_nbrs))
            else:
                break         
        
        return walk

    # attr节点带偏置随机游走
    def gen_attr_walk(self,startnode):
        
        G = self.G1
        walklen = self.attrwalklen
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [startnode]

        while len(walk) < walklen:
                cur = walk[-1]
                cur_nbrs = sorted(G.neighbors(cur))
                if len(cur_nbrs) > 0:
                    if len(walk) == 1:
                        walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                    else:
                        prev = walk[-2]
                        next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
                        alias_edges[(prev, cur)][1])]
                        walk.append(next)
                else:
                    break        

        return walk

    # 产生所有node序列和attr序列
    def gen_walks(self):
        
        G = self.G
        walknum = self.walknum
        print("random walk...")
        node_walks = []
        nodes = list(G.nodes())
        for _ in range(walknum):
            random.shuffle(nodes)
            for node in nodes:
                node_walks.append(self.gen_node_walk(node))

        G1 = self.G1
        walknum = self.walknum
        attr_walks = []
        attrs = list(G1.nodes())
        for _ in range(walknum):
            random.shuffle(attrs)
            for attr in attrs:
                attr_walks.append(self.gen_attr_walk(attr))
        
        return node_walks,attr_walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G1
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G1

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}
		
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]