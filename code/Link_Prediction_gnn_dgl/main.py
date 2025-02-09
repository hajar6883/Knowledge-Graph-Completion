#data loading, model training, and evaluation

import torch
import torch.nn as nn
import dgl
from dgl.dataloading import GraphDataLoader
from models import LinkPredict, RGCN,GAT
from utils import ArtworksGraph, SubgraphIterator, calc_mrr

def get_subset_g(g, mask, num_rels, bidirected=False):
    src, dst = g.edges()
    sub_src = src[mask]
    sub_dst = dst[mask]
    sub_rel = g.edata["etype"][mask]

    if bidirected:
        sub_src, sub_dst = torch.cat([sub_src, sub_dst]), torch.cat(
            [sub_dst, sub_src]
        )
        sub_rel = torch.cat([sub_rel, sub_rel + num_rels])

    sub_g = dgl.graph((sub_src, sub_dst), num_nodes=g.num_nodes())
    sub_g.edata[dgl.ETYPE] = sub_rel
    return sub_g

def train(
    dataloader,
    test_g,
    test_nids,
    val_mask,
    triplets,
    device,
    model_state_file,
    model,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    best_mrr = 0
    for epoch, batch_data in enumerate(dataloader):  # single graph batch
        model.train()
        g, train_nids, edges, labels = batch_data
        g = g.to(device)
        g = dgl.add_self_loop(g)

        train_nids = train_nids.to(device)
        edges = edges.to(device)
        labels = labels.to(device)

        embed = model(g, train_nids)
        loss = model.get_loss(embed, edges, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # clip gradients
        optimizer.step()
        print(
            "Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f}".format(
                epoch, loss.item(), best_mrr
            )
        )
        if (epoch + 1) %  100 == 0:
            # perform validation on CPU because full graph is too large
            # model = model.cpu()
            model.eval()
            
            test_g = test_g.to(device)
            test_nids = test_nids.to(device)
            # triplets = triplets.to(device)
            embed = model(test_g, test_nids)
         
            mrr, hits_at_1, hits_at_3, hits_at_10 = calc_mrr(
                embed, model.w_relation, val_mask, triplets, batch_size=500
            )
            #***
            # save best model
            if best_mrr < mrr:
                best_mrr = mrr
                torch.save(
                    {"state_dict": model.state_dict(), "epoch": epoch},
                    model_state_file,
                )
            # model = model.to(device)
            print(
                f"Epoch {epoch + 1:04d} | Loss {loss.item():.4f} | "
                f"MRR {mrr:.4f} | Hits@1 {hits_at_1:.4f} | "
                f"Hits@3 {hits_at_3:.4f} | Hits@10 {hits_at_10:.4f}"
            )#***

def main():

    data = ArtworksGraph()
    g = data[0]

    num_nodes = g.num_nodes()
    num_rels = data.num_rels
    train_g = get_subset_g(g, g.edata["train_mask"], num_rels)
    test_g = get_subset_g(g, g.edata["train_mask"], num_rels, bidirected=True)
    test_g.edata["norm"] = dgl.norm_by_dst(test_g).unsqueeze(-1)
    test_nids = torch.arange(0, num_nodes)
    val_mask = g.edata["val_mask"]
    test_mask = g.edata["test_mask"]
    subg_iter = SubgraphIterator(train_g, num_rels)
    dataloader = GraphDataLoader(
        subg_iter, batch_size=1, collate_fn=lambda x: x[0]
    )

    src, dst = g.edges()
    triplets = torch.stack([src, g.edata["etype"], dst], dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name='RGCN'
    model = LinkPredict(RGCN, num_nodes, num_rels).to(device)

    model_state_file = f"{model_name}_model_state_Joconde.pth"


    train(
        dataloader,
        test_g,
        test_nids,
        val_mask,
        triplets,
        device,
        model_state_file,
        model,
    )

    print("Testing...")
    checkpoint = torch.load(model_state_file)
    # model = model.cpu()  # test on CPU
    model.eval()
    model.load_state_dict(checkpoint["state_dict"])

    test_g = test_g.to(device)
    test_nids = test_nids.to(device)

    embed = model(test_g, test_nids)
    best_mrr, hits_at_1, hits_at_3, hits_at_10 = calc_mrr(
    embed, model.w_relation, test_mask, triplets, batch_size=500)

    print(
            f"Best MRR {best_mrr:.4f} | Hits@1 {hits_at_1:.4f} | "
            f"Hits@3 {hits_at_3:.4f} | Hits@10 {hits_at_10:.4f} | "
            f"achieved using the epoch {checkpoint['epoch']:04d}"
        )
    
if __name__ == "__main__":
    main()
