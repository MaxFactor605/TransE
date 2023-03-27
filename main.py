import torch
import os
from dataset import WordNetTriplets
from model import TransE, TransELoss
from torch.utils.data import DataLoader


def train(model, triplets, corrupted_triplets, loss_fn, optim, num_of_epochs = 10, 
          val_triplets = None, val_corrupted_triplets = None, test_triplets = None, test_corrupted_triplets = None, weight_path='./weights'):
    model.train(True)

    try:
        model.load_state_dict(torch.load(weight_path)) # try to load weights, if not presented, continue
    except FileNotFoundError:
        pass


    for epoch in range(num_of_epochs):
        epoch_loss = 0
        for batch, batch_corrupted in zip(triplets, corrupted_triplets):
            
            optim.zero_grad()

            model.normalize_entities()

            batch_emb = model(*batch)
            batch_corrupted_emb = model(*batch_corrupted)

            loss = loss_fn(batch_emb, batch_corrupted_emb)
            epoch_loss += loss.item()
            loss.backward()

            optim.step()

        if(val_triplets and val_corrupted_triplets):
            val_loss = 0

            for batch, batch_corrupted in zip(val_triplets, val_corrupted_triplets):
                with torch.no_grad():
                    batch_emb = model(*batch)
                    batch_corrupted_emb = model(*batch_corrupted)
                    loss = loss_fn(batch_emb, batch_corrupted_emb)
                    val_loss += loss.item()

            print(f"Epoch {epoch}:\n\tavr_loss {epoch_loss/len(triplets)}\tval_loss {val_loss/len(val_triplets)}")

        else:
            print(f"Epoch {epoch}:\n\tavrg_loss {epoch_loss/len(triplets)}")

    if(test_triplets and test_corrupted_triplets):
        test_loss = 0

        for batch, batch_corrupted in zip(test_triplets, test_corrupted_triplets):
                with torch.no_grad():
                    batch_emb = model(*batch)
                    batch_corrupted_emb = model(*batch_corrupted)
                    loss = loss_fn(batch_emb, batch_corrupted_emb)
                    test_loss += loss.item()

        print(f"Training finished!\n\tTest_loss : {test_loss/len(test_triplets)}")
    else:
        print(f"Training finished!")

    torch.save(model.state_dict(), weight_path)
    print("Model saved!")


if __name__ == '__main__':
    triplets = WordNetTriplets()
    corrupted_triplets = WordNetTriplets(corrupted=1)
    val_tripltes = WordNetTriplets(data_type = 'val')
    val_corrupted_triplets = WordNetTriplets(corrupted=1, data_type='val')

    triplets_dl = DataLoader(triplets, 100)
    corrupted_triplets_dl = DataLoader(corrupted_triplets, 100)
    val_tripltes_dl = DataLoader(val_tripltes, 100)
    val_corrupted_triplets_dl = DataLoader(val_corrupted_triplets, 100)
    
    model = TransE(triplets.num_nodes, triplets.num_relations, 20)
    loss = TransELoss(2)
    
    optimizer = torch.optim.SGD(model.parameters(), 0.01)

    train(model, triplets_dl, corrupted_triplets_dl, loss, optimizer, 100, val_tripltes_dl, val_corrupted_triplets_dl)