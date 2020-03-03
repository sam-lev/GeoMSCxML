from __future__ import print_function
import json
from networkx.readwrite import json_graph
import numpy as np

from topoml.graphsage.utils import load_data
from topoml.graphsage.utils import format_data
import networkx as nx
from argparse import ArgumentParser

''' To evaluate the embeddings, we run a logistic regression.
Run this script after running unsupervised training.
Baseline of using features-only can be run by setting data_dir as 'feat'
Example:
  python eval_scripts/ppi_eval.py ../data/ppi unsup-ppi/n2v_big_0.000010 test
python  eval_scripts/ppi_eval.py (dataset_dir) ../example_data (embed_dir) ../unsup-example_data/graphsage_mean_small_0.000010 (setting) test
python  eval_scripts/ppi_eval.py example_data unsup-example_data/graphsage_mean_small_0.000010  test

python  eval_scripts/ppi_eval.py (dataset_dir) ../json_graphs (embed_dir) /Users/multivax/Documents/PhD/Research/topologicalComputing-Pascucci/TopoML/graphSage/GraphSAGE/unsup-json_graphs/graphsage_mean_small_0.000010 (setting) test

python  eval_scripts/ppi_eval.py ../json_graphs /Users/multivax/Documents/PhD/Research/topologicalComputing-Pascucci/TopoML/graphSage/GraphSAGE/unsup-json_graphs/graphsage_mean_small_0.000010 test
'''
import os

class LinearRegression:
    def __init__(self, test_path = None,  test_prefix = None
                 , trained_path = None, trained_prefix = None
                 ,  embedding_path = None, inference_embedding_path=None, setting = 'test'
                 , G=None, MSCGNN_infer=None, MSCGNN=None,
                 features = None, labels=None, num_neg = None,
                 id_map=None, walks = None, with_features=False):

        self.trained_path = trained_path
        self.test_path = test_path
        self.test_prefix = test_prefix
        self.trained_prefix = trained_prefix
        self.embedding_path = embedding_path
        self.inference_embedding_path = inference_embedding_path
        self.G = G
        self.labels = labels
        self.id_map = id_map
        self.num_neg = num_neg
        self.with_features=with_features

        if MSCGNN_infer is not None:
            self.mscgnn_infer = MSCGNN_infer
            self.G_infer = MSCGNN_infer.G
            self.inference_node_ids = MSCGNN_infer.node_id
            #self.inference_id_map = self.inference_node_ids
            #if isinstance(self.G_infer.nodes()[0], int):
            #    conversion = lambda n: int(n)
            #else:
            #    conversion = lambda n: n
            #self.inference_id_map = {conversion(k): int(v) for k, v in self.inference_id_map.items()}
            self.inference_node_classes = MSCGNN_infer.node_classes
            self.inference_features = MSCGNN_infer.features
            self.inference_positive_arcs = MSCGNN_infer.positive_arcs
            self.inference_negative_arcs = MSCGNN_infer.negative_arcs
            self.inference_msc = MSCGNN_infer.msc
            self.G_infer, self.inference_features, self.inference_id_map\
                ,self.inference_walks, self.inference_class_map, self.inference_negative_samples\
                ,self.inference_positive_samples = format_data(dual=self.G_infer,features=self.inference_features
                                                               ,node_id=self.inference_node_ids, id_map=self.inference_node_ids
                                                               ,node_classes=self.inference_node_classes
                                                               ,train_or_test='', scheme_required=True, load_walks=False)
        if MSCGNN is not None:
            self.mscgnn = MSCGNN
            self.G = MSCGNN.G
            self.node_ids = MSCGNN.node_id
            self.node_classes = MSCGNN.node_classes
            self.feats = MSCGNN.features
            self.number_positive_samples = MSCGNN.positive_arcs
            self.num_neg = MSCGNN.negative_arcs
            self.msc = MSCGNN.msc
            self.G, self.feats, self.id_map, self.walks\
                ,self.class_map, self.number_negative_samples\
                ,self.number_positive_samples = format_data(dual=self.G, features=self.feats, node_id=self.node_ids
                                                            ,id_map=self.node_ids, node_classes=self.node_classes
                                                            ,train_or_test='', scheme_required=True, load_walks=False)
        else:
            self.feats = features
            self.labels = labels
            self.num_neg = num_neg
            self.number_positive_samples = num_neg
            self.id_map = id_map
            self.walks = walks
        
        self.model_name = ''
        if trained_prefix is not None:
            if len(trained_prefix.split("/")) >1:
                self.model_name =  trained_prefix.split("/")[-1]
        else:
            self.model_name = trained_prefix
        
        self.train_prefix = self.model_name
        self.label_node_predictions =  True #args.label_node_predictions
        self.setting = setting             #args.setting
        
    def run_regression(self, train_embeds=None, train_labels=None, test_embeds=None, test_labels=None, test_ids = None, test_graph = None, embeds=None, id_map=None):
        np.random.seed(1)
        from sklearn.linear_model import SGDClassifier
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import f1_score
        from sklearn.multioutput import MultiOutputClassifier
        dummy = MultiOutputClassifier(DummyClassifier())
        dummy.fit(train_embeds, train_labels)
        log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=2)

        log.fit(train_embeds, train_labels)
        prediction = log.predict(test_embeds)
        f1 = 0
        for i in range(test_labels.shape[1]):
            print("F1 score", f1_score(test_labels[:,i], log.predict(test_embeds)[:,i], average="binary"))
        for i in range(test_labels.shape[1]):
            print("Random baseline F1 score", f1_score(test_labels[:,i], dummy.predict(test_embeds)[:,i], average="micro"))

        if test_graph:
            
            #for id, pred in zip(test_ids, prediction.tolist()):
            #for i in test_ids:    
            #    test_graph.node[i]["prediction"] = prediction[i,:]
                #print( test_graph.node[id])

            predictions = {}
            for id in test_ids:
                pred =log.predict(embeds[[id_map[id]]])[0]
                test_graph.node[id]["prediction"] = [int(pred[0]),int(pred[1])]
            if self.mscgnn_infer is not None:
                for id, arc in zip(test_ids, self.mscgnn_infer.arcs):
                    pred = log.predict(embeds[[id_map[id]]])[0]
                    arc.label_accuracy =  float(pred[1])#[int(pred[0]),int(pred[1])]

            pred_path = os.path.join(self.embedding_path.split("/")[:-1][0], self.embedding_path.split("/")[:-1][1],self.embedding_path.split("/")[:-1][2],'predicted_graph-G.json')
            
            if not os.path.exists(  pred_path):
                open( pred_path, 'w').close()
            with open( pred_path, 'w') as graph_file:
                write_form = json_graph.node_link_data(test_graph)
                json.dump(write_form, graph_file)
            print("Prediction written to: ", pred_path)
            if self.mscgnn_infer is not None:
                return self.mscgnn_infer

    def run(self):
        
        if self.trained_path:
            print("TRAIN PATH:  ", self.trained_path)
            self.G, self.feats, self.id_map, self.walks, self.labels, self.num_neg, self.number_positive_samples = load_data(self.trained_path, load_walks=False, scheme_required = True, train_or_test='train')
        elif self.G:
            G = self.G
            feats = self.feats
            walks = self.walks
            labels = self.labels
            number_negative_samples = self.num_neg
            id_map = self.id_map

        G, feats, id_map, walks, labels, number_negative_samples, number_positive_samples = self.G, self.feats, self.id_map, self.walks, self.labels, self.num_neg,self.number_positive_samples
        #G = json_graph.node_link_graph(json.load(open(self.trained_path+"-G.json")))
        if self.test_path is not None and self.G_infer is None:
            G_infer = json_graph.node_link_graph(json.load(open(self.test_path+"-G.json")))
            self.G_infer, self.inference_features, self.inference_id_map, self.inference_walks, self.inference_labels, self.inference_negative_samples, self.inference_positive_samples = load_data(self.test_path, load_walks=False, scheme_required=True, train_or_test='train')
        else:
            G_infer = self.G_infer
        #labels = json.load(open(self.trained_path+"-class_map.json"))
        labels = {int(i):l for i, l in self.labels.items()}#iteritems()} #for python3
        #labels_test = json.load(open(self.test_path+"-class_map.json"))
       

        train_ids = [n for n in G.nodes()]# if  G.node[n]['train']]
        
        test_ids = [n for n in G_infer.nodes()]# if 'label' in G.node[n]]#(G.node[n]['test'] or G.node[n]['val'] or G.node[n]['train'])]

        train_labels = np.array([labels[i] for i in train_ids])
        
        if train_labels.ndim == 1:
            train_labels = np.expand_dims(train_labels, 1)
            
        test_labels = np.array([labels[i] for i in test_ids])

        print("Running trained model: ", self.trained_prefix)
        print("Total training samples: ", train_labels.shape)
        print("Total testing samples: ", test_labels.shape)
        

        if self.trained_prefix == "feat" or self.with_features == True:
            print("\n", "Using only features (not embedding).","\n")
            if self.feats is None:
                feats = np.load(self.trained_path+"-feats.npy")
            else:
                feats = self.feats
    
            ## Logistic gets thrown off by big counts, so log transform num comments and score
            feats[:,0] = np.log(feats[:,0]+1.0)
            feats[:,1] = np.log(feats[:,1]-min(np.min(feats[:,1]), -1))
            if not self.feats:
                id_map = json.load(open(self.trained_path+"-id_map.json"))
            feat_id_map = {int(id):val for id,val in id_map.items()}#iteritems()}
            train_feats = feats[[id_map[id] for id in train_ids]]
            feats_test = self.inference_features[[self.inference_id_map[id] for id in test_ids]]

            feats_test[:,0] = np.log(feats_test[:,0]+1.0)
            feats_test[:,1] = np.log(feats_test[:,1]-min(np.min(feats_test[:,1]), -1))
            #feat_id_map_test = json.load(open(self.test_path+"-id_map.json"))
            #feat_id_map_test = {int(id):val for id,val in feat_id_map_test.items()}#iteritems()}
            test_feats_test = feats[[id_map[id] for id in test_ids]]

            print("\n","Running regression...","\n")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(train_feats)
            train_feats = scaler.transform(train_feats)
            test_feats = scaler.transform(test_feats_test)
            #if not label_node_predictions:
            #    run_regression(train_feats, train_labels, test_feats, test_labels)
            #else:
            self.run_regression(train_feats, train_labels, test_feats, test_labels, test_ids=test_ids, test_graph=G_infer)
        else:
            print("EMBEDDING: ", self.embedding_path)
            embeds = np.load(self.embedding_path + "/val.npy")
            id_map = {}
            with open(self.embedding_path+ "/val.txt") as fp:
                for i, line in enumerate(fp):
                    id_map[int(line.strip())] = i
            train_embeds = embeds[[id_map[id] for id in train_ids]]

            if False and self.inference_embedding_path and not self.inference_id_map:
                inference_embeds = np.load(self.inference_embedding_path + "/val.npy")
                inference_id_map = {}
                with open(self.inference_embedding_path + "/val.txt") as fp:
                    for i, line in enumerate(fp):
                        inference_id_map[int(line.strip())] = i
                test_embeds = inference_embeds[[inference_id_map[id] for id in test_ids]]
            else:
                test_embeds = embeds[[self.id_map[id] for id in test_ids]]
            print("Running regression..")
            if not self.label_node_predictions:
                self.run_regression(train_embeds, train_labels, test_embeds, test_labels)
            else:
                self.run_regression(train_embeds, train_labels, test_embeds, test_labels, test_ids=test_ids, test_graph=G_infer, embeds=embeds, id_map=id_map)
        if self.mscgnn_infer is not None:
            return self.mscgnn_infer