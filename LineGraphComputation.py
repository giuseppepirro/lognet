import numpy as np
import pandas as pd
from typing import Tuple, Set
import itertools
from joblib import Parallel, delayed
import multiprocessing
import os
from time import time
import csv as csv

num_cores = multiprocessing.cpu_count()


def add_edges(source_node_ID, all_triples):
    column=0
    mask = np.in1d(all_triples[:, column], [source_node_ID])
    ##we need to add edges between the ids of the triple in triple_IDs
    triple_IDs = np.argwhere(mask == True)
    edgesAdded: Set[Tuple]=[]

    for x in itertools.product(triple_IDs, triple_IDs):
        if x[0]!=x[1]:
            edge: Tuple[str, str] = (x[0].item()), (x[1].item())
            edgesAdded.append(edge)
    return edgesAdded

def write_triple_line_graph(results,KG,basepath):
    csv_f = basepath + KG + '/line_graph.csv'
    with open(csv_f,'w') as csv_file:
        writer = csv.writer(csv_file)
        for res in results:
            for r in res:
                writer.writerow(r)

def computeLineGraph(alltriples,numEntities,KG,basepath):
    start = time()
    results = Parallel(n_jobs=num_cores + 2, verbose=10, prefer="threads")(delayed(add_edges)(subjectID,alltriples) for subjectID in range(0,numEntities+1))
    print("Time to compute line graph {:.6f} s".format(time() - start))
    write_triple_line_graph(results,KG,basepath)

def main():
    KG = "FB15K-NEG"
    basepath = "./DATA/"
    train = basepath + KG + "/triples_train.csv"
    test = basepath + KG + "/triples_test.csv"

    t_train=np.genfromtxt(train,usecols = (0, 1, 2,3),delimiter=",",dtype=np.int)
    t_test=np.genfromtxt(test,usecols = (0, 1, 2,3),delimiter=",",dtype=np.int)
    all_triples=np.append(t_train,t_test,axis=0)

    #all_triples contain triples in the form s,o,p we need to swap o and p
    all_triples = all_triples[:, [0, 2,1,3]]

    d = dict(enumerate((all_triples[:, [0,2,1]]), 0))
    csv_f = basepath + KG + '/triples2ids.csv'
    with open(csv_f, 'w') as csv_file:
        print(d, file=csv_file)

    #exclude the labels
    all_triples =all_triples[:, :-1]
    num_entities = np.amax(all_triples)
    print("Max entity id=",num_entities)
    computeLineGraph(alltriples=all_triples,numEntities=num_entities,KG=KG,basepath=basepath)


if __name__ == "__main__":
    main()

