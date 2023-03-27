# TransE


Model for translating embeddings for graph nodes and edges. 
Trained on WordNet18RR dataset

Based on this [article](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)

## Files

* model.py - implementation of model and loss classes
* dataset.py - implementation of dataset class for converting torch_geometric WordNet18RR dataset into set of triplets
* main.py - implementation of train function and main file for launching training
* weights - file with weights after a 100 epoch (delete to generate new or change path for train function in main.py)
