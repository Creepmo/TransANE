#coding:utf-8
from graph import Graph
from train import Train,Eval
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument("--data",nargs="?",default="Haverford",help="Swarthmore Haverford")
    parser.add_argument("--dimensions",type=int,default=256,help="the dimension of embeddings")
    parser.add_argument("--nodewalklen",type=int,default=40,help="the walk length of nodes")
    parser.add_argument("--nodewinsize",type=int,default=5,help="the window size of nodes")
    parser.add_argument("--attrwalklen",type=int,default=40,help="the walk length of attributes")
    parser.add_argument("--attrwinsize",type=int,default=5,help="the window size of attributes")
    parser.add_argument("--walknum",type=int,default=10,help="the number of walks")	
    parser.add_argument('--p', type=float, default=0.5,help="return hyperparameter")
    parser.add_argument('--q', type=float, default=2.0,help="inout hyperparameter")

    return parser.parse_args()

def main(args):
    args.inputpath = "./data/%s/"%args.data
    args.embpath = "emb/%s.emb"%args.data
    graph = Graph(args)
    graph.pre_processing()
    graph.read_graph()
    graph.preprocess_transition_probs()
    node_walks,attr_walks = graph.gen_walks()
    train = Train(args,node_walks,attr_walks)
    train.tf_train_step()
    eval = Eval(args)
    eval.node_classify()
    eval.network_visualization(args.data)

if __name__=="__main__":
    args = parse_args()
    main(args)