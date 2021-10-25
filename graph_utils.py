from cupy_utils import *
import networkx as nx

def topk_mean(m, k, inplace=False, is_self=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    if is_self:
        k += 1
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        if i == 0 and is_self:
            pass
        else:
            ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

def calc_csls_sim(x, y, k=10, is_self=False):
    batch_size = 1024
    trg_size = y.shape[0]
    src_size = x.shape[0]
    dtype = 'float32'
    
    xp = get_array_module(x)

    simfwd = xp.empty((batch_size, trg_size), dtype=dtype)
    simbwd = xp.empty((batch_size, src_size), dtype=dtype)

    knn_sim_fwd = xp.zeros(src_size, dtype=dtype)
    knn_sim_bwd = xp.zeros(trg_size, dtype=dtype)
    
    # y->x方向的
    for i in range(0, trg_size, simbwd.shape[0]):
        j = min(i + simbwd.shape[0], trg_size)
        y[i:j].dot(x.T, out=simbwd[:j-i])
        # batch_size * 1
        knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=k, inplace=True, is_self=is_self)

    # x->y方向的
    for i in range(0, src_size, simfwd.shape[0]):
        j = min(i + simfwd.shape[0], src_size)
        x[i:j].dot(y.T, out=simfwd[:j-i])
        knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=k, inplace=True, is_self=is_self)

    #print(knn_sim_fwd.sum() / knn_sim_fwd.shape[0])
    #print(knn_sim_bwd.sum() / knn_sim_bwd.shape[0])
    #print(knn_sim_bwd[:20])
    #print(knn_sim_fwd[:20])

    csls_sim_matrix = x.dot(y.T) * 2 - knn_sim_fwd[:, xp.newaxis] - knn_sim_bwd
    #print(csls_sim_matrix[0, :20])

    return csls_sim_matrix



    


class word_graph:

    def __init__(self, embeddings, csls_k=10, min_edge_weight=0, max_neighbor_number=0):
        self.embs = embeddings
        self.N = self.embs.shape[0]
        self.edges_list, self.A = self.calc_adj_matrix(embeddings, csls_k, min_edge_weight, max_neighbor_number)
        

    def calc_adj_matrix(self, embeddings, csls_k, min_edge_weight, max_neighbor_number):
        xp = get_array_module(embeddings)

        sim_matrix = calc_csls_sim(embeddings, embeddings, csls_k, True)
        
        #sim_matrix = embeddings.dot(embeddings.T)
        sim_matrix = sim_matrix * (1 - xp.identity(sim_matrix.shape[0]))


        # filter weight
        low_weight_mask = sim_matrix >= min_edge_weight
        sim_matrix = sim_matrix * low_weight_mask
        
        # filter neighbor number
        if max_neighbor_number > 0:
            sorted_sim_tmp = xp.sort(sim_matrix)
            # max_neighbor_number+1 for ignore self
            kth_sim_value_tmp = sorted_sim_tmp[:, -(max_neighbor_number+1)]
            assert kth_sim_value_tmp.shape[0] == sim_matrix.shape[0]

            for i in range(sim_matrix.shape[0]):
                neighbor_mask = sim_matrix[i] >= kth_sim_value_tmp[i]
                sim_matrix[i] = sim_matrix[i] * neighbor_mask

        edges_list = xp.argwhere(sim_matrix >= min_edge_weight).tolist()
        edges_list = list(set([(s, t) for s, t in edges_list]))
        print("edge count: ", len(edges_list))

        return edges_list, sim_matrix

    def get_cliques_list(self, min_cliques_number=2):
        G = nx.Graph()
        G.add_edges_from(self.edges_list)
        res = nx.find_cliques(G)
        cliques_list = [item for item in res if len(item) > min_cliques_number]
        return cliques_list

    def get_clique_center_lists(self, min_cliques_number=2):
        xp = get_array_module(self.embs)

        # calc cliques
        self.cliques_list = self.get_cliques_list(min_cliques_number)

        # get center word index
        clique_center_lists = []
        for clique_word_index_list in self.cliques_list:
            clique_embedding = self.embs[clique_word_index_list]
            avg_clique_embedding = xp.mean(clique_embedding, axis=0)
            clique_sim_score = clique_embedding.dot(avg_clique_embedding.T)
            center_word_index = clique_word_index_list[int(xp.argmax(clique_sim_score))]
            clique_center_lists.append(center_word_index)

        #clique_center_lists = sorted(clique_center_lists)
        return clique_center_lists





            


