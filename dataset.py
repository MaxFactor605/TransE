import random
from torch.utils.data import Dataset
from torch_geometric.datasets import WordNet18RR


class WordNetTriplets(Dataset):
    def __init__(self, path="./WordNet18RR", corrupted=0, data_type='train'):
        self.wordNet = WordNet18RR(path).get(0)

        self.num_nodes = self.wordNet.num_nodes
        self.num_relations = self.wordNet.num_edge_types
        self.corrupted = corrupted # We'll implement 2 diffrent kinds of sets of triplets, one with real relation and corrupted one
        
        # Number of elements in train, val and test dataset are taken from raw WordNet18RR data
        if(data_type == 'train'):
            self.heads = self.wordNet.edge_index[0][:86836]
            self.relations = self.wordNet.edge_type[:86836]
            self.tails = self.wordNet.edge_index[1][:86836]
        elif(data_type == 'val' or data_type == 'validation'):
            self.heads = self.wordNet.edge_index[0][86836:89970]
            self.relations = self.wordNet.edge_type[86836:89970]
            self.tails = self.wordNet.edge_index[1][86836:89970]
        elif(data_type == 'test'):
            self.heads = self.wordNet.edge_index[0][89970:]
            self.relations = self.wordNet.edge_type[89970:]
            self.tails = self.wordNet.edge_index[1][89970:]


    def __getitem__(self, idx):
        if self.corrupted:
            return self.get_corrupted_triplet(idx)
        
        return self.heads[idx], self.relations[idx], self.tails[idx]


    def __len__(self):
        return len(self.heads)


    def get_corrupted_triplet(self, idx):
        neg_tail = self.tails[random.randint(0, len(self.tails)-1)]

        return self.heads[idx], self.relations[idx], neg_tail

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = WordNetTriplets()
    dataset_corrupted = WordNetTriplets(corrupted=1)

    dataloader = DataLoader(dataset, 10)
    dataloader_corrupted = DataLoader(dataset_corrupted, 10)

    i = 0
    for batch in dataloader:
        print(batch)
        i += 1
        if(i == 15):
            break
    
    print("===================")
    for batch in dataloader_corrupted:
        print(batch)
        i += 1
        if(i == 30):
            break