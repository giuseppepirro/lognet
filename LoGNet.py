import numpy as np
import time
import torch as th
import torch.nn as nn
import dgl.nn as dglnn
from KGLineDataset import KGLineDataset
import math
import sklearn as skit


class LoGNet(nn.Module):
    def __init__(self, embedding_dim, entity_count, predicate_count,weight_plausibility):
        super(LoGNet, self).__init__()
        self.entity_count = entity_count
        self.predicate_count = predicate_count

        print("Entity count=",self.entity_count)
        print("Predicate count=",self.predicate_count)

        self.num_triples = triple_count
        self.embedding_dim = embedding_dim

        #Initialize the embeddings of all entities and predicates
        self.entities_emb = self._init_enitity_emb()
        self.relations_emb = self._init_relation_emb()
        ###############################
        self.ranking_loss=nn.MarginRankingLoss(margin=1.0)

        self.bigru_local_embeddings=nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True, num_layers=1,
                                        bidirectional=True)
        self.gat_global_embeddings=dglnn.GATConv(in_feats=embedding_dim,num_heads=1,out_feats=embedding_dim,attn_drop=0.5)


        self.normalization_layer = nn.LayerNorm(normalized_shape=embedding_dim*6) #should be embedding_dim*6

        self.scale_emb_dim=nn.Linear(in_features=6*embedding_dim, out_features=embedding_dim)

        self.weight_plausibility=weight_plausibility
        self.margin=1

    def forward(self, kg_triples, labels, line_graph,nodes_line_graph, mask):
        #Trasform each triple to its representation via the embeddings that have been previously initialized
        # for each s,p,o we should obtain [ [entity_emb(s)], [pred_emb(p)], [entity_emb(o)]]
        kg_triples=np.array(kg_triples)
        batch_size=kg_triples.shape[0]
        subjects = th.tensor(kg_triples[:, 0])
        predicates = th.tensor(kg_triples[:, 1])
        objects = th.tensor(kg_triples[:, 2])

        #Indexes of triples labeled as True
        indexes_pos_examples=np.where(labels == 1)[0]

        #Indexes of triples labeled as False
        indexes_neg_examples=np.where(labels== -1)[0]

        eS=self.entities_emb(subjects)
        eO=self.entities_emb(objects)
        eP=self.entities_emb(predicates)

        vectorized_triples=th.cat((eS,eP,eO),dim=1)

        vectorized_triples=vectorized_triples.view(batch_size,3, self.embedding_dim)

        #h contains [[subj_fwd subj_backw][pred_fwd pred_backw][obj_fwd obj_backw]]
        h,triple_hidden_state= self.bigru_local_embeddings(vectorized_triples)


        subjects_bigru = h[:, :1, ]  ##all subjects
        predicates_bigru = h[:, 1:2, ]  ##all predicates
        objects_bigru = h[:, 2:3, ]  ##all objects

        localPlausibility = subjects_bigru+ predicates_bigru - objects_bigru
        localPlausibility=th.linalg.norm(localPlausibility,dim=2)

        localPlausibility=th.sigmoid(localPlausibility)


        all_emb_concatenate = (th.cat((subjects_bigru, predicates_bigru, objects_bigru), dim=1)).view(batch_size, 1, embedding_dim* 6)
        h=all_emb_concatenate

        h=self.normalization_layer(h)

        h=self.scale_emb_dim(h)

        #Set local embeddings as features of the line graph
        line_graph.nodes[nodes_line_graph].data['features']= h


        z = self.gat_global_embeddings(line_graph, h)

        globalPlausibilityT = th.linalg.norm(z,dim=2)
        globalPlausibilityT =th.sigmoid(globalPlausibilityT)
        globalPlausibility=globalPlausibilityT-localPlausibility

        plausibility_score=(self.weight_plausibility * localPlausibility) + globalPlausibility

        plausibility_score =plausibility_score[mask]

        negative_plausibility=plausibility_score[indexes_neg_examples]

        positive_plausibility=plausibility_score[indexes_pos_examples]

        assert len(positive_plausibility)==len(negative_plausibility)

        target = th.ones_like(positive_plausibility)

        return self.ranking_loss(input1=positive_plausibility,input2=negative_plausibility,target=target), z, plausibility_score

    def _init_enitity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.embedding_dim,
                                    padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.embedding_dim)
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb


    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.predicate_count + 1,
                                     embedding_dim=self.embedding_dim,
                                     padding_idx=self.predicate_count)
        uniform_range = 6 / np.sqrt(self.embedding_dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relations_emb

def AUCEvaluator(model, triples, line_graph, labels,nodes_line,mask):
    print("Number of test triples=", len(labels_test))
    model.eval()
    with th.no_grad():
        loss,  z, plausibility_score = model(triples,  labels, line_graph,nodes_line,mask)
        auc=skit.metrics.roc_auc_score(labels,plausibility_score)
        return auc


### TEST
KG="FB15K-NEG"
dataset = KGLineDataset(raw_dir="../DATA/"+KG)

line_graph=dataset.line_graph
triples=dataset.triples
triple_count=len(dataset.triples)
entity_count=dataset.entity_count
predicate_count=dataset.predicate_count

batch_size=triple_count
total_samples = len(dataset.labels)
n_iterations = math.ceil(total_samples / batch_size)


line_graph=dataset.line_graph

train_mask = th.gt(dataset.train_mask,0)
test_mask = th.gt(dataset.test_mask,0)
len_train=len(th.where(train_mask==True)[0])
len_test=len(th.where(test_mask==True)[0])
labels = dataset.labels
labels_train = labels[0:len_train]
labels_test = labels[len_train:]


# Training hyperparameters
weight_decay = 5e-4
n_epochs = 100
lr = 1e-3
###

embedding_dim=50
input_feature_size=50
output_dim=1
weight_plausibility=0.3

nodes_line_graph=range(0,triple_count)

model=LoGNet(embedding_dim, entity_count, predicate_count,weight_plausibility)

optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

print("Start training..........")
for epoch in range(n_epochs):
        # Set the model in the training mode.
        model.train()
    #   # forward
        loss, z, plausibility_score = model(triples, labels_train,line_graph,nodes_line_graph,train_mask)
        optimizer.zero_grad()
    #   #backward
        loss.backward()
        optimizer.step()
        print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))

print("Start testing......")
auc = AUCEvaluator(model, triples=triples, line_graph=line_graph, labels=labels_test,nodes_line=nodes_line_graph,mask=test_mask)
print("Test Accuracy {:.4f}".format(auc))