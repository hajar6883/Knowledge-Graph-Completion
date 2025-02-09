import torch
import dgl
import numpy as np
import pandas as pd
import tqdm
from dgl.data import DGLDataset

"""class that construct hetergeneous graph from Joconde Knowledge graph"""
class ArtworksGraph(DGLDataset):
    def __init__(self):
        super().__init__(name="Artworks")

    def process(self):
        """!!!!!need to include the preprocessing file that reconstruct this dfs from .nt triples  """

        artworks = pd.read_csv("artworks_augmented.csv")
        relations = pd.read_csv("props.csv")

        edges_src = torch.from_numpy(relations["Src"].values)
        edges_dst = torch.from_numpy(relations["Dest"].values)
        edge_types = pd.Categorical(relations["Prop"]).codes
        edge_types = torch.from_numpy(edge_types.astype('int64'))

        self.num_rels = len(pd.Categorical(relations["Prop"]).categories)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=artworks.shape[0])
        self.graph.edata["etype"] = edge_types

        n_edges = relations.shape[0]
        n_train = int(n_edges * 0.6)
        n_val = int(n_edges * 0.2)

        train_mask = torch.zeros(n_edges, dtype=torch.bool)
        val_mask = torch.zeros(n_edges, dtype=torch.bool)
        test_mask = torch.zeros(n_edges, dtype=torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True

        self.graph.edata["train_mask"] = train_mask
        self.graph.edata["val_mask"] = val_mask
        self.graph.edata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

# Samplers
class GlobalUniform:
    def __init__(self, g, sample_size):
        self.sample_size = sample_size
        self.eids = np.arange(g.num_edges())

    def sample(self):
        return torch.from_numpy(np.random.choice(self.eids, self.sample_size))


class NegativeSampler:
    def __init__(self, k=10):
        self.k = k

    def sample(self, pos_samples, num_nodes):
        batch_size = len(pos_samples)
        neg_batch_size = batch_size * self.k
        neg_samples = np.tile(pos_samples, (self.k, 1))

        values = np.random.randint(num_nodes, size=neg_batch_size)
        choices = np.random.uniform(size=neg_batch_size)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]
        samples = np.concatenate((pos_samples, neg_samples))

        labels = np.zeros(batch_size * (self.k + 1), dtype=np.float32)
        labels[:batch_size] = 1
        
        return torch.from_numpy(samples), torch.from_numpy(labels)


class SubgraphIterator:
    def __init__(self, g, num_rels, sample_size=30000, num_epochs=1000):
        self.g = g
        self.num_rels = num_rels
        self.sample_size = sample_size
        self.num_epochs = num_epochs
        self.pos_sampler = GlobalUniform(g, sample_size)
        self.neg_sampler = NegativeSampler()

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, i):
        eids = self.pos_sampler.sample()
        src, dst = self.g.find_edges(eids)
        src, dst = src.numpy(), dst.numpy()
        rel = self.g.edata[dgl.ETYPE][eids].numpy()

        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        num_nodes = len(uniq_v)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_data = np.stack((src, rel, dst)).transpose()

        samples, labels = self.neg_sampler.sample(relabeled_data, num_nodes)

        chosen_ids = np.random.choice(
            np.arange(self.sample_size),
            size=int(self.sample_size / 2),
            replace=False,
        )
        src = src[chosen_ids]
        dst = dst[chosen_ids]
        rel = rel[chosen_ids]
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + self.num_rels))
        sub_g = dgl.graph((src, dst), num_nodes=num_nodes)
        sub_g.edata[dgl.ETYPE] = torch.from_numpy(rel)
        sub_g.edata["norm"] = dgl.norm_by_dst(sub_g).unsqueeze(-1)
        uniq_v = torch.from_numpy(uniq_v).view(-1).long()

        return sub_g, uniq_v, samples, labels
    
def filter( #Generates a list of candidate nodes for link prediction while filtering out known true triplets
    triplets_to_filter, target_s, target_r, target_o, num_nodes, filter_o=True
):
    """Get candidate heads or tails to score"""
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    # Add the ground truth node first
    if filter_o:
        candidate_nodes = [target_o]
    else:
        candidate_nodes = [target_s]
    for e in range(num_nodes):
        triplet = (
            (target_s, target_r, e) if filter_o else (e, target_r, target_o)
        )
        # Do not consider a node if it leads to a real triplet
        if triplet not in triplets_to_filter:
            candidate_nodes.append(e)
    return torch.LongTensor(candidate_nodes)

# Evaluation Metrics
def perturb_and_get_filtered_rank(
    emb, w, s, r, o, test_size, triplets_to_filter, filter_o=True
):
    """Perturb subject or object in the triplets and return ranks and Hits@1,3,10"""
    num_nodes = emb.shape[0]
    ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    
    for idx in tqdm.tqdm(range(test_size), desc="Evaluate"):
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        candidate_nodes = filter(
            triplets_to_filter,
            target_s,
            target_r,
            target_o,
            num_nodes,
            filter_o=filter_o,
        )
        if filter_o:
            emb_s = emb[target_s]
            emb_o = emb[candidate_nodes]
        else:
            emb_s = emb[candidate_nodes]
            emb_o = emb[target_o]
        target_idx = 0
        emb_r = w[target_r]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))

        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)
        
        # Compute Hits@K
        if rank < 1:
            hits_at_1 += 1
        if rank < 3:
            hits_at_3 += 1
        if rank < 10:
            hits_at_10 += 1
            
    return torch.LongTensor(ranks), hits_at_1, hits_at_3, hits_at_10


def calc_mrr(emb, w, mask, triplets_to_filter, batch_size=100, filter=True): 
    """added more rank metrics"""
    with torch.no_grad():
        test_triplets = triplets_to_filter[mask]
        s, r, o = test_triplets[:, 0], test_triplets[:, 1], test_triplets[:, 2]
        test_size = len(s)
        triplets_to_filter = {
            tuple(triplet) for triplet in triplets_to_filter.tolist()
        }
        
        ranks_s, hits_at_1_s, hits_at_3_s, hits_at_10_s = perturb_and_get_filtered_rank(
            emb, w, s, r, o, test_size, triplets_to_filter, filter_o=False
        )
        ranks_o, hits_at_1_o, hits_at_3_o, hits_at_10_o = perturb_and_get_filtered_rank(
            emb, w, s, r, o, test_size, triplets_to_filter
        )
        
        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed
        
        mrr = torch.mean(1.0 / ranks.float()).item()
        hits_at_1 = (hits_at_1_s + hits_at_1_o) / (2 * test_size)
        hits_at_3 = (hits_at_3_s + hits_at_3_o) / (2 * test_size)
        hits_at_10 = (hits_at_10_s + hits_at_10_o) / (2 * test_size)
        
    return mrr, hits_at_1, hits_at_3, hits_at_10
