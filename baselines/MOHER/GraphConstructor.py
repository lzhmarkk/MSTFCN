import torch


class GraphConstructor:
    def __init__(self, device, n_mix, n_nodes, adj_mx, gamma, beta, subgraph_size, poi_feat=None):
        self.device = device
        self.n_mix = n_mix
        self.n_nodes = n_nodes
        self.subgraph_size = subgraph_size
        self.gamma = gamma
        self.beta = beta
        self.poi_feat = poi_feat

        self.sim = [torch.tensor(adj_mx).float()]
        if poi_feat is not None:
            poi_sim = torch.zeros(n_nodes, n_nodes, dtype=torch.float)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    sim = torch.cosine_similarity(poi_feat[i], poi_feat[j])
                    if sim >= self.beta:
                        poi_sim[i][j] = sim
            poi_sim = (poi_sim - poi_sim.mean()) / poi_sim.std()
            self.sim.append(poi_sim)

        self.graphs, self.closest_neighbors, self.closest_neighbors_weight = self.__construct_graphs()

    def __find_neighbors(self, sim_matrix, target):
        # ignore time since our geography does not change
        neighbors = []
        # BFS
        queue = [(target, 1)]
        visited = torch.zeros(self.n_nodes)
        while len(neighbors) < self.subgraph_size and len(queue) > 0:
            i, sim = queue.pop(0)
            visited[i] = sim
            neighbors.append(i)

            for j in range(self.n_nodes):
                if visited[j] == 0:
                    v = sim_matrix[i, j].item()
                    if v > 0:
                        queue.append((j, v))

            sorted(queue, key=lambda e: e[1], reverse=True)
        value, idx = torch.topk(visited, largest=True, sorted=True, k=self.subgraph_size)
        length = (value != 0).sum()
        return idx, length

    def __construct_graphs(self):
        res = [], [], []
        for sim in self.sim:
            nodes_neighbor = []
            nodes_neighbor_weight = []
            mask = torch.zeros_like(sim)
            for tgt in range(self.n_nodes):
                neighbors, n_neighbors = self.__find_neighbors(sim, tgt)
                neighbors = neighbors[:n_neighbors]
                mask[neighbors, tgt] = 1 / n_neighbors
                nodes_neighbor.append(neighbors[:2])
                nodes_neighbor_weight.append(sim[neighbors[:2], tgt])

            sim = sim * mask
            res[0].append(sim.to(self.device))
            res[1].append(torch.stack(nodes_neighbor, 0).to(self.device))
            res[2].append(torch.stack(nodes_neighbor_weight, 0).to(self.device))

        return res

    def get_graphs(self):
        assert len(self.graphs) == len(self.closest_neighbors) == len(self.closest_neighbors_weight)
        return self.graphs, self.closest_neighbors, self.closest_neighbors_weight
