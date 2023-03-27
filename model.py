import torch
import math


class TransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim):
        super(TransE, self).__init__()
        self.emb_dim = emb_dim

        self.entity_embeddings = torch.nn.Embedding(num_entities, emb_dim)
        self.relation_embeddings = torch.nn.Embedding(num_relations, emb_dim)

        # Embeddings init
        torch.nn.init.uniform_(self.entity_embeddings.weight.data, -6/math.sqrt(emb_dim), 6/math.sqrt(emb_dim))
        torch.nn.init.uniform_(self.relation_embeddings.weight.data, -6/math.sqrt(emb_dim), 6/math.sqrt(emb_dim))

        # Normalize relation embeddings 
        self.relation_embeddings.weight.data = torch.nn.functional.normalize(self.relation_embeddings.weight.data, p=1, dim = 1)

    def forward(self, head, rel, tail):
        head_emb = self.entity_embeddings(head)
        rel_emb = self.relation_embeddings(rel)
        tail_emb = self.entity_embeddings(tail)

        return head_emb, rel_emb, tail_emb

    # Method for normalization of entities embeddings
    def normalize_entities(self):
        self.entity_embeddings.weight.data = torch.nn.functional.normalize(self.entity_embeddings.weight.data, p = 1, dim =1)
       

class TransELoss(torch.nn.Module):
    def __init__(self, margin, norm_order=1):
        super(TransELoss, self).__init__()
        self.margin = margin
        self.norm_order = norm_order
        self.criterion = torch.nn.MarginRankingLoss(margin)

    def forward(self, triplet, corrupted_triplet):
        head = triplet[0]
        rel = triplet[1]
        tail = triplet[2]
        tail_corupted = corrupted_triplet[2]
        
        diff_triplet = torch.norm(head+rel-tail, p=self.norm_order, dim = 1)
        diff_triplet_corrupted = torch.norm(head+rel-tail_corupted, p=self.norm_order, dim = 1)
        
        return self.criterion(diff_triplet, diff_triplet_corrupted, -torch.Tensor([-1]))
    

if __name__ == '__main__':
    from dataset import WordNetTriplets
    from torch.utils.data import DataLoader
    dataset = WordNetTriplets()
    dataset_corrupted = WordNetTriplets(corrupted=1)
    loss = TransELoss(5)
    dataloader = DataLoader(dataset, 15)
    dataloader_corrupted = DataLoader(dataset_corrupted, 15)
    
    print(dataset.num_relations)
    model = TransE(dataset.num_nodes, dataset.num_relations, 10)


    batch = next(iter(dataloader))
    batch_corrupted = next(iter(dataloader_corrupted))

    batch_emb = model(*batch)
    batch_corrupted_emb = model(*batch_corrupted)

    print(batch_emb[0].shape, batch_emb[1].shape, batch_emb[2].shape)
    print(batch_corrupted_emb[0].shape, batch_corrupted_emb[1].shape, batch_corrupted_emb[2].shape)
    print(loss(batch_emb, batch_corrupted_emb))