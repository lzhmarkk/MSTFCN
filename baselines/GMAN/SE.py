import os
import torch
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from baselines.GMAN.Graph import Graph

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
iter = 1000


def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())

    return G


def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    print("Training word2vec")
    model = Word2Vec(walks, vector_size=dimensions, window=10, min_count=0, sg=1,
                     workers=8, epochs=iter)
    print("Train word2vec done")
    model.wv.save_word2vec_format(output_file)
    return


def generateSE(Adj_file, SE_file):
    nx_G = read_graph(Adj_file)
    G = Graph(nx_G, is_directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    learn_embeddings(walks, dimensions, SE_file)


def load_se_file(adj_mx, adj_file, se_file):
    if not os.path.exists(se_file):
        nodes_num = adj_mx.shape[0]
        with open(adj_file, mode='w') as f:
            for i in range(nodes_num):
                for j in range(nodes_num):
                    dis = adj_mx[i][j]
                    f.write(str(i) + " " + str(j) + " " + str(dis) + "\n")
        generateSE(adj_file, se_file)
        os.remove(adj_file)

    with open(se_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    return SE
