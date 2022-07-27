import os
import time
import tqdm
import dgl.sampling
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from utils_LoGNet import load_model, set_random_seed
import configure_LoGNet


######################################
#@TODO initial edge weight normalization
def normalize_edge_weights(graph, device, num_ew_channels):
    degs = graph.in_degrees().float()
    degs = torch.clamp(degs, min=1)
    norm = torch.pow(degs, 0.5)
    norm = norm.to(args["device"])
    graph.ndata["norm"] = norm.unsqueeze(1)
    graph.apply_edges(fn.e_div_u("feat", "norm", "feat"))
    graph.apply_edges(fn.e_div_v("feat", "norm", "feat"))
    for channel in range(num_ew_channels):
        graph.edata["feat_" + str(channel)] = graph.edata["feat"][:, channel : channel + 1]
######################################

#######
#@TODO: a train EPOCH
def run_a_train_epoch(train_dataloader, model, criterion, optimizer, evaluator):
    model.train()
    #train indexes are handled by the train_loader!
    with tqdm.tqdm(train_dataloader) as tq:  # this statement refers to the progress bar.
        #@TODO: mfgs depends on the number of layers!
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            blocks = [blk.to(args["device"]) for blk in mfgs]
            input_features = blocks[0].srcdata
            ##
            labels = blocks[-1].dstdata['labels']
            logits = model(blocks, input_features)  # getting predictions.
            ##
            #### Compute the loss
            loss = criterion(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.data.item()
            labels = labels.cpu().numpy()
            preds = logits.cpu().detach().numpy()
    return loss, evaluator.eval({"y_true": labels, "y_pred": preds})["rocauc"]


def run_an_eval_epoch(valid_dataloader, splitted_idx,model, evaluator):
    model.eval()
    predictions = []
    labels = []
    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
    train_score = evaluator.eval({"y_true": labels[splitted_idx["train"]], "y_pred": predictions[splitted_idx["train"]]})
    val_score = evaluator.eval({"y_true": labels[splitted_idx["valid"]], "y_pred": predictions[splitted_idx["valid"]]})
    test_score = evaluator.eval({"y_true": labels[splitted_idx["test"]], "y_pred": predictions[splitted_idx["test"]]})

    return train_score["rocauc"], val_score["rocauc"], test_score["rocauc"]


def main(args):
    print(args)
    if args["rand_seed"] > -1:
        set_random_seed(args["rand_seed"])
    dataset = DglNodePropPredDataset(name=args["dataset"])

    #print(dataset.meta_info)
    splitted_idx = dataset.get_idx_split()
    graph = dataset.graph[0]
    #Node labels
    graph.ndata["labels"] = dataset.labels.float().to(args["device"])
    #Edge features
    graph.edata["feat"] = graph.edata["feat"].float().to(args["device"])

    if args["ewnorm"] == "both":
        print("Symmetric normalization of edge weights by degree")
        normalize_edge_weights(graph, args["device"], args["num_ew_channels"])
    elif args["ewnorm"] == "none":
        print("Not normalizing edge weights")

        ## Add a new type of edge for each feature channel
        for channel in range(args["num_ew_channels"]):
            graph.edata["feat_" + str(channel)] = graph.edata["feat"][:, channel : channel + 1]

    #Load the model
    model = load_model(args).to(args["device"])
    optimizer = Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    min_lr = 1e-3
    scheduler = ReduceLROnPlateau(optimizer, "max", factor=0.7, patience=100, verbose=True, min_lr=min_lr)
    #### Loss function
    criterion = nn.BCEWithLogitsLoss()
    ##Evaluator
    evaluator = Evaluator(args["dataset"])
    ####

    dur = []
    best_val_score = 0.0
    num_patient_epochs = 0
    model_folder = "./saved_models/"
    model_path = model_folder + str(args["exp_name"]) + "_" + str(args["postfix"])
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    neighbour_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(4*num_layer)
    ####

    batch_size = 250
    train_dataloader = dgl.dataloading.NodeDataLoader(
        graph,  # The graph
        splitted_idx["train"],  # The node IDs to iterate over in minibatches
        neighbour_sampler,  # The neighbor neighbour_sampler
        device=args["device"],  # Put the sampled MFGs on CPU or GPU
        batch_size=batch_size,  # Batch size
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
    )
    ####################################################################################

    #TRAINING
    for epoch in range(1, args["num_epochs"] + 1):
        if epoch >= 3:
            t0 = time.time()
        loss, train_score = run_a_train_epoch(train_dataloader, model, criterion, optimizer, evaluator)
        if epoch >= 3:
            dur.append(time.time() - t0)
            avg_time = np.mean(dur)
        else:
            avg_time = None

######################################################################
        ###Evaluation
        valid_dataloader = dgl.dataloading.NodeDataLoader(
            graph, splitted_idx, neighbour_sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            device=args["device"]
        )
        train_score, val_score, test_score = run_an_eval_epoch(valid_dataloader, model, evaluator)
        scheduler.step(val_score)

        # Early stop
        if val_score > best_val_score:
            torch.save(model.state_dict(), model_path)
            best_val_score = val_score
            num_patient_epochs = 0
        else:
            num_patient_epochs += 1

        print(
            "Epoch {:d}, loss {:.4f}, train score {:.4f}, "
            "val score {:.4f}, avg time {}, num patient epochs {:d}".format(
                epoch, loss, train_score, val_score, avg_time, num_patient_epochs
            )
        )

        if num_patient_epochs == args["patience"]:
            break

    model.load_state_dict(torch.load(model_path))
    train_score, val_score, test_score = run_an_eval_epoch(valid_dataloader, model, evaluator)
    print("Train score {:.4f}".format(train_score))
    print("Valid score {:.4f}".format(val_score))
    print("Test score {:.4f}".format(test_score))
    #
    with open("results.txt", "w") as f:
        f.write("loss {:.4f}\n".format(loss))
        f.write("Best validation rocauc {:.4f}\n".format(best_val_score))
        f.write("Test rocauc {:.4f}\n".format(test_score))
    print(args)


if __name__ == "__main__":
    #CPU version
    args={"dataset":"ogbn-proteins",'model':'LogNet',"rand_seed":22,"device":'cpu',"num_ew_channels":8,"ewnorm":"both","postfix":"test","exp_name":"test-g
    ",
    'num_ew_channels': 8,                                                                                                                                                            'num_epochs': 2000,
    'in_feats': 1,
    'hidden_feats': 10,
    'out_feats': 112,
    'n_layers': 3,
    'lr': 2e-2,
    'weight_decay': 0,
    'patience': 1000,
    'dropout': 0.2,
    'aggr_mode': 'sum',  ## 'sum' or 'concat' for the aggregation across channels
    'ewnorm': 'both',
    'residual': True
    }
    #args["exp_name"] = "_".join([args["model"], args["dataset"]])
    #args.update(configure.get_exp_configure(args))
    #GPU version
    #args={"dataset":"ogbn-proteins",'model':'MWE-GCN',"rand_seed":22,"device":'cuda:0',"num_ew_channels":8,"ewnorm":"both","postfix":"prova","exp_name":"prova-giu"}

    main(args)
