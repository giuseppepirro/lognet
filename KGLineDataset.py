from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
import pandas as pd
import torch as th
import numpy as np
import dgl
import os

class KGLineDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.
    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 url=None,
                 name="KG_LINE",
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(KGLineDataset, self).__init__(name=name+"_cache",
                                            url=url,
                                            raw_dir=raw_dir,
                                            save_dir=save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

    def download(self):
        pass

    def process(self):
        ## BEGIN: Line Graph data
        line_graph_weighted_edges= pd.read_csv(os.path.join(self.raw_dir, 'line_graph.csv'),header=None)
        line_graph_weighted_edges.columns = ['s', 'd']
        src = line_graph_weighted_edges['s']
        dst = line_graph_weighted_edges['d']
        line_graph_dgl = dgl.graph((src, dst))
        line_graph_dgl = dgl.add_self_loop(line_graph_dgl)
        print("Number of nodes line graph (equals to the number of triples) =",line_graph_dgl.num_nodes())
        line_graph_dgl.ndata['features'] = th.rand(line_graph_dgl.num_nodes(),1, 50) #Default feature=50
        self._features = line_graph_dgl.ndata['features']
        ## END: Line Graph data

        ## BEGIN: Knowledge Graph data
        t_train = np.genfromtxt(os.path.join(self.raw_dir, 'triples_train.csv'), usecols=(0, 1, 2,3), delimiter=",",dtype=np.int)
        t_test = np.genfromtxt(os.path.join(self.raw_dir, 'triples_test.csv'), usecols=(0, 1, 2,3), delimiter=",",dtype=np.int)
        all_triples = np.append(t_train, t_test, axis=0)
        #labels are in the 3rd column
        labels=all_triples[:, [3]]
        # exclude the labels 3rd column (last)
        all_triples = all_triples[:, :-1]
        ## END: Knowledge Graph data

        ##Set the split train and test
        labels_train1 = np.full(len(t_train), True)
        labels_train2 = np.full(len(t_test), False)
        labels_train=np.append(labels_train1,labels_train2)

        labels_test1 = np.full(len(t_train), False)
        labels_test2 = np.full(len(t_test), True)
        labels_test=np.append(labels_test1,labels_test2)

        line_graph_dgl.ndata['train_mask'] = th.BoolTensor(labels_train)
        line_graph_dgl.ndata['test_mask'] = th.BoolTensor(labels_test)

        # Set triples, labels and line graph in the KGLineDataset
        self._triples = all_triples
        self._labels = labels
        self._num_labels = 2  # This is for fact checking:either a fact is true (value 1) or false (value 0)

        line_graph_dgl.ndata['labels'] = th.tensor(labels)

        self._line_graph = line_graph_dgl

        #Dictionaries
        entities_ids_df = pd.read_csv(os.path.join(self.raw_dir, 'entities2ids.csv'))['entity']
        predicates_ids_df = pd.read_csv(os.path.join(self.raw_dir, 'preds2ids.csv'))['pred']
        self._entities_dict = entities_ids_df.to_dict()
        self._predicates_dict = predicates_ids_df.to_dict()


    def __getitem__(self, idx):
        return self._triples[idx],self.labels[idx],idx #nodi del line graph

    def __len__(self):
        return len(self._triples)

    def __iter__(self):
        return iter(self.data)


    def save(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name() + '.bin')
        info_path = os.path.join(self.save_path,self.save_name())
        save_graphs(str(graph_path), self._line_graph)
        save_info(str(info_path + '_triples.pkl'), self._triples)
        save_info(str(info_path + '_entity.pkl'), self._entities_dict)
        save_info(str(info_path + '_pred.pkl'), self._predicates_dict)

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name() + '.bin')
        info_path = os.path.join(self.save_path, self.save_name())
        graphs = load_graphs(str(graph_path))
        self._triples = load_info(str(info_path + '_triples.pkl'))
        self._line_graph = graphs[0][0]
        self._entities_dict = load_info(str(info_path + '_entity.pkl'))
        self._predicates_dict= load_info(str(info_path + '_pred.pkl'))

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name() + '.bin')
        info_path = os.path.join(self.save_path, self.save_name())
        is_graph_loaded = os.path.isfile(graph_path)
        is_triples_dict_loaded = os.path.isfile(str(info_path + '_triples.pkl'))
        is_entity_dict_loaded = os.path.isfile(str(info_path + '_entity.pkl'))
        is_pred_dict_loaded = os.path.isfile(str(info_path + '_pred.pkl'))
        return (is_graph_loaded and is_entity_dict_loaded and is_pred_dict_loaded and is_triples_dict_loaded)

    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def triple_count(self):
        return len(self.triples)

    @property
    def triples(self):
        return self._triples

    @property
    def entity_count(self):
        return len(self.entities_dictionary.values())

    @property
    def entities_dictionary(self):
        return self._entities_dict

    @property
    def predicate_count(self):
        return len(self.predicates_dictionary.values())

    @property
    def predicates_dictionary(self):
        return (self._predicates_dict)

    @property
    def line_graph(self):
        return self._line_graph

    @property
    def train_mask(self):
        return self._line_graph.ndata['train_mask']

    @property
    def val_mask(self):
        return  self._line_graph.ndata['val_mask']

    @property
    def test_mask(self):
        return self._line_graph.ndata['test_mask']

    @property
    def labels(self):
        return self._line_graph.ndata['labels']

    @property
    def features(self):
        return self._line_graph.ndata['features']

    @property
    def num_labels(self):
        return self._num_labels

def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask