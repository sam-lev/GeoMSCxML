
import os
from time import time
import datetime
import numpy as np

# 3rd party imports
import json
from networkx.readwrite import json_graph
import networkx as nx



#from topoml.graphsage import gnn
from topoml.graphsage.gnn import unsupervised as unsupervised_gnn
from topoml.eval_scripts import LinearRegression

# import class for labeling MSC from supervised segmentation
# import class for computing MSC or Geometric MSC segmenting images

# GNN imports (local)
from topoml.graphsage.utils import format_data

# Local application imports
""" ! need to call tracer with param passed giving user option 
       to select"""
""" give option to compute features from given selection or
    pre defined graph"""

from topoml.ui.ArcSelector import ArcSelector
#To build MSC and select arcs
from topoml.ArcNeuronTracer import ArcNeuronTracer
from topoml.ml.MSCSample import MSCSample
from topoml.topology.geometric_msc import GeoMSC
from topoml.ui.LocalSetup import LocalSetup
from topoml.graphsage.utils import random_walk_embedding

class MSCGNN:
    def __init__(self, msc=None, geomsc=None, G=None, gnn=None, select=False):
        self.msc = msc if msc is not None else geomsc
        self.geomsc = geomsc if geomsc is not None else msc
        self.G = G
        self.select = select
        self.positive_arcs = None
        self.negative_arcs = None
        self.features = None
        self.node_id = None
        self.node_classes = None
        self.gnn = gnn if gnn is not None else None
        if self.msc is not None:
            self.assign_msc(self.msc)

    def assign_msc(self, msc):
        self.msc = msc
        self.geomsc = msc
        self.arcs = msc.arcs
        self.arc_dict = msc.arc_dict
        self.arc_map = msc.arc_map
        self.nodes = msc.nodes

    def msc_feature_graph(self,  image
                  , validation_samples, validation_hops, test_samples, test_hops
                  , accuracy_threshold
                  ,msc=None, load=False, X=None, Y=None, write_json_graph_path=False, name=None):
        if load:
            msc = GeoMSC()
            msc = msc.read_from_file(fname_base=load, labeled=True)
        if msc is not None:
            self.msc = self.assign_msc(msc)
        if X is None:
            X = image.shape[1]
            Y = image.shape[2]

        print("%% computing subgraph train, val, test split")

        msc = MSCSample(msc=self.msc, image=image)
        self.G,\
        self.node_id,\
        self.node_classes,\
        self.features = msc.msc_subgraph_splits(X=X, Y=Y, validation_samples=validation_samples, validation_hops=validation_hops
                                            , test_samples=test_samples, test_hops=test_hops, accuracy_threshold=accuracy_threshold)
        self.positive_arcs = msc.positive_arcs
        self.negative_arcs = msc.negative_arcs
        self.msc = msc

        print("% subgraph split complete ")

        if write_json_graph_path:

            print('.writing graph family data')
            s = time()
            if not os.path.exists(os.path.join(write_json_graph_path, 'json_graphs')):
                os.mkdir(os.path.join(write_json_graph_path, 'json_graphs'))
            graph_file_path = os.path.join(write_json_graph_path, 'json_graphs')
            # group_name = 'left'
            for graph_data, f_name in zip([self.G, self.node_id, self.node_classes, self.features],
                                          [name + '-G', name + '-id_map', name + '-class_map']):

                if not os.path.exists(os.path.join(graph_file_path, f_name + '.json')):
                    open(os.path.join(graph_file_path, f_name + '.json'), 'w').close()
                with open(os.path.join(graph_file_path, f_name + '.json'), 'w') as graph_file:
                    json.dump(graph_data, graph_file)

            if not os.path.exists(os.path.join(graph_file_path, name + '-feats.npy')):
                open(os.path.join(graph_file_path, name + '-feats.npy'), 'w').close()
            np.save(os.path.join(graph_file_path, name + '-feats.npy'), self.features)

            f = time()
            print('graph family written in & w/ prefix ', graph_file_path + '/' + name, '(', f - s, ')')


    def select_msc(self, train_or_test=''
                   ,selector=None
                   ,tracer=None
                   ,group_name='unnamed'
                   ,bounds = []
                   ,write_json = False
                   ,write_msc = False
                   ,load_msc = 'test_ridge_arcs'
                   ,cvt=tuple()
                   ,val_count=10):
        """

        @param train_or_test: construct GNN json graph for testing or training
        @type  train_or_test: string

        @param group_name: name to write json (for gnn) and csv (for msc) files
        @type  group_name: string

        @param bounds: (optional) [[xmin, xmax],[ymin,ymax]] if subset of image/msc desired. Default []
        @type bounds: list
        
        @param write_json: (optional) write selected / computed msc to JSON file for GNN. Default: False
        @type  write_json: bool

        @param write_msc: write selected / computed msc to file to CSV 
        @type  write_json: bool

        @param load_msc: (optional) file name to load pre-computed/selected msc
        @type  laod_msc: string

        @rtype:   No Return
        @return:  No Return.
        """

        cwd = './'
        
        if load_msc and not cvt:
            
            print('loading msc')
            s = time()
            
            msc_w_path =  os.path.join(cwd,'data','msc_graphs')
            group_msc_file =  os.path.join(msc_w_path,load_msc+'-MSC.csv')
            try:
                open( group_msc_file, 'r').close()
            except:
                print("No Morse Smale File Found.")
            print("MSC FILE: ", group_msc_file)
            selector.load_arcs(group_msc_file)
            in_arcs = selector.in_arcs
            out_arcs = selector.out_arcs
            gt_in_arcs, gt_out_arcs = [], []
            f = time()
            print('loaded msc(',f-s,')')
            
        if bounds and not load_msc:
            print('using bounds')
            #selector.load_arcs(group_msc_file)
            xmin, ymin = bounds[0][0], bounds[1][0]
            xmax, ymax = bounds[0][1], bounds[1][1]
            selector.cull_selection([xmin, xmax, ymin, ymax])
            in_arcs, out_arcs, out_pixels = selector.launch_ui(xlims=[xmin, xmax], ylims=[ymin, ymax])
            gt_in_arcs, gt_out_arcs = [],[]
            
        elif not bounds and not load_msc and not cvt: 
            in_arcs, out_arcs, out_pixels = tracer.do_selection()
            gt_in_arcs, gt_out_arcs = [] , []
            
        elif cvt:
            n_samples = cvt[0]
            hops = cvt[1]
            s = time()
            #selector = ArcSelector(tracer.raw_image, tracer.msc, valley=False)
            tracer = ArcNeuronTracer("./data/max_diadem.png", blur_sigma=2, persistence=1, valley=False)
            print('collecting ground truth graph')
            
            msc_w_path =  os.path.join(cwd,'data','msc_graphs')
            group_msc_file =  os.path.join(msc_w_path,load_msc+'-MSC.csv')
            selector = ArcSelector(tracer.raw_image, tracer.msc, valley=False)

            try:
                open( group_msc_file, 'r').close()
            except:
                print("No Morse Smale File Found.")
                
            selector.load_arcs(group_msc_file)
            
            gt_in_arcs = selector.in_arcs
            gt_out_arcs = selector.out_arcs
            #tracer.do_selection(selector)
            print('finished full ground truth graph. Selecting training set with CVT sampling')

            in_arcs, out_arcs = selector.sample_selection(count=n_samples, rings=hops, seed=123)
            val_in_arcs, val_out_arcs = selector.sample_selection(count=1, rings=int(hops/2), seed=111)
            
            f = time()
            print('Finished cvt sampling in '+str(f-s))

            #cvt_indices = selector.in_arcs + selector.out_arcs
            #in_arcs = selector.in_arcs
            #out_arcs = selector.out_arcs

            print('size selected in arcs.', len(in_arcs))
            print('size selected out arcs.', len(out_arcs))
            print('size val. in.', len(val_in_arcs))
            print('size val. out.', len(val_out_arcs))

        self.positive_arcs, self.negative_arcs = in_arcs, out_arcs 

        
        print('Begining labeled graph formating...')
        s = time()
        
        G, node_id\
            , node_classes\
            , node_features = tracer.create_graphsage_input(self.positive_arcs
                                                            , self.negative_arcs
                                                            , groundtruth_positive_arcs=gt_in_arcs
                                                            , groundtruth_negative_arcs= gt_out_arcs
                                                            , val_count=val_count
                                                            , test_positive_arcs= val_in_arcs
                                                            , test_negative_arcs=val_out_arcs)


        self.G = G
        self.msc = tracer.msc
        self.features = node_features
        self.node_id = node_id
        self.node_classes = node_classes
        
        f= time()
        print('Finished graph formatting in (',f-s,')')
        


        if write_msc:
            
            print('writing msc')
            
            msc_w_path =  os.path.join(cwd,'data','msc_graphs')
            if not load_msc:
                group_msc_file =  os.path.join(msc_w_path,group_name+'-MSC.csv')
            else:
                group_msc_file =  os.path.join(msc_w_path,load_msc+'-MSC.csv')
            if not os.path.exists(group_msc_file):
                open( group_msc_file, 'w').close()
            selector.load_arcs(group_msc_file)
            selector.save_arcs(group_msc_file,'a')

            print('msc hath been wrote')


        if write_json:

            print('.writing graph family data')
            s = time()
            
            graph_file_path =  os.path.join(cwd,'data','json_graphs')
            #group_name = 'left'
            for graph_data, f_name in zip([G,node_id,node_classes,node_features],[group_name+'-G',group_name+'-id_map',group_name+'-class_map']):
                
                if not os.path.exists( os.path.join(graph_file_path,f_name+'.json') ):
                    open( os.path.join(graph_file_path,f_name+'.json'), 'w').close()
                with open( os.path.join(graph_file_path,f_name+'.json'), 'w') as graph_file:
                    json.dump(graph_data, graph_file)

            if not os.path.exists( os.path.join(graph_file_path,group_name+'-feats.npy') ):
                open( os.path.join(graph_file_path,group_name+'-feats.npy'), 'w').close()
            np.save(  os.path.join(graph_file_path,group_name+'-feats.npy'), node_features)

            f= time()
            print('graph family written in & w/ prefix ', graph_file_path+'/'+group_name, '(',f-s,')')


    def unsupervised(self, aggregator = None, slurm = False):
        self.gnn = unsupervised_gnn(aggregator = aggregator, slurm = slurm)
    """Train the GNN on dual of msc with arc features in nodes"""
    
    def train(self,generate_embedding=False, graph = None, features = None, node_id = None, node_classes = None,
              normalize = True, train_prefix = '',  embedding_name = '', load_walks=False,
              num_neg = None, learning_rate=None, epochs=200, weight_decay=0.01,
              polarity=6, depth=3):
        """
        ### trains on --train_prefix (adds -G.json to param passed when searching for file)
        ### uses --model aggregator for training
        ### takes --max_total_steps
        ### number validation iterations --validate_iter
        ### saves trained embedding in 'log-dir/'--model_name
        @param train_group_name: directory name and group name to graph info files e.g. 'test_dual_graphs/test' 
        @type train_group_name: string
        """
  
        
        if self.G and not train_prefix:#and not cvt_train:
            
            G,feats,id_map\
                , walks, class_map\
                , number_negative_samples\
                , number_positive_samples = format_data(dual=self.G
                                                        ,features=self.features
                                                        ,node_id=self.node_id
                                                        ,id_map=self.node_id
                                                        ,node_classes=self.node_classes
                                                        ,train_or_test = ''
                                                        ,scheme_required = True
                                                        ,load_walks=load_walks)
            
            self.gnn.train( G=G,
                            learning_rate = learning_rate,
                            load_walks=load_walks,
                            number_negative_samples = len(self.negative_arcs),
                            number_positive_samples = number_positive_samples,
                            feats = feats,
                            id_map=id_map,
                            class_map=class_map,
                            embedding_file_out = embedding_name,
                            epochs=epochs,
                            weight_decay = weight_decay,
                            polarity=polarity,
                            depth=depth)


        
        if train_prefix:
            print("loading")
            cwd = './'
            walks = load_walks
            train_path =  os.path.join(cwd,'data','json_graphs',train_prefix)
            
            if self.negative_arcs:
                num_neg=len(self.negative_arcs)

                
            if not embedding_name:
                embedding_name = datetime.datetime.now()
                
            self.gnn.train(train_prefix=train_path,
                           learning_rate = learning_rate,
                           load_walks=load_walks,
                           number_negative_samples = num_neg,
                           embedding_file_out = embedding_name,
                           epochs=epochs,
                           weight_decay=weight_decay,
                           polarity=polarity,
                           depth=depth)

    def embed_inference_msc(self, inference_mscgnn, persistence, blur, inference_embedding_file=None,walk_embedding_file=None):
        aggregator = ['graphsage_maxpool', 'gcn', 'graphsage_seq', 'graphsage_maxpool'
            , 'graphsage_meanpool', 'graphsage_seq', 'n2v'][4]

        print('... Beginning training with aggregator: ', aggregator)

        # Random walks used to determine node pairs for unsupervised loss.
        # to make a random walk collection use ./topoml/graphsage/utils.py
        # example run: python3 topoml/graphsage/utils.py ./data/json_graphs/test_ridge_arcs-G.json ./data/random_walks/full_msc_n-1_k-40
        ##load_walks='full_msc_n-1_k-40'
        ##load_walks='full_msc_n-1_k-50_Dp-10_nW-100'
        print('... Generating Random Walk Neighborhoods for Node Co-Occurance')

        random_walk_embedding(inference_mscgnn.G, walk_length=10, number_walks=5, out_file=walk_embedding_file)


        # hyper-param for gnn
        learning_rate = 2e-3
        polarity = 25
        weight_decay = 0.01
        epochs = 10
        depth = 3
        batch_size = len(inference_mscgnn.nodes)
        self.train(generate_embedding=True,  embedding_name=inference_embedding_file, load_walks=walk_embedding_file
                     , learning_rate=learning_rate, epochs=epochs
                     , weight_decay=weight_decay, polarity=polarity, depth=depth)

    """Perform classification using learned graph representation from GNN"""
    def classify(self, MSCGNN_infer = None, test_prefix = None,  trained_prefix=None, embedding_prefix=None, aggregator='graphsage_mean', learning_rate = None):
        cwd = './'
        #embedding_path =  os.path.join(cwd,'log-dir',embedding_prefix+'-unsup-json_graphs','graphsage_mean_small_'+'0.100000')
        embedding_p = embedding_prefix+'-unsup-json_graphs'+'/'+aggregator+'_'+'big'
        embedding_p += ("_{lr:0.6f}").format(lr=learning_rate)
        if test_prefix is not None and trained_prefix is not None and not self.G:
            trained_p = os.path.join(cwd,'data','json_graphs',trained_prefix)
            test_p =  os.path.join(cwd,'data','json_graphs',test_prefix)
            trained_prfx = trained_prefix
            test_prfx = test_prefix
            LinearRegression(test_path = test_p, MSCGNN_infer = MSCGNN_infer, test_prefix = test_prfx, trained_path = trained_p, trained_prefix = trained_prfx, embedding_path = os.path.join(cwd, 'log-dir',embedding_p)).run()
            
        elif self.G:
             G,feats,id_map, walks, class_map, number_negative_samples, number_positive_samples = format_data(dual=self.G, features=self.features, node_id=self.node_id, id_map=self.node_id, node_classes=self.node_classes, train_or_test = '', scheme_required = True, load_walks=False)
             
             LinearRegression(G=G,
                              MSCGNN_infer=MSCGNN_infer,
                              features = feats,
                              labels=class_map,
                              num_neg = len(self.negative_arcs),
                              id_map = id_map,
                              embedding_path = os.path.join(cwd, 'log-dir',embedding_p)).run()

        
        
        
        #eval_scripts/test_eval.py json_graphs/left_train_thro json_graphs/right_test_thro log-dir/left_thro_rnd2-unsup-left_train_thro/graphsage_mean_small_0.000010 left_train right_test True test

    def show_gnn_classification(self,pred_graph_prefix = None, msc=None, G=None, gs_G =None, msc_G = None, train_view = False):
        if pred_graph_prefix:
            cwd = './'
            pred_G_path =  os.path.join(cwd,'log-dir',pred_graph_prefix+'-unsup-json_graphs','predicted_graph-G.json')
            G_pred = json_graph.node_link_graph(json.load(open(pred_G_path)))
            
        else:
            G_pred = G
        tracer = ArcNeuronTracer("./data/max_diadem.png", blur_sigma=2, persistence=1, valley=False)
        tracer.show_classified_graph(G=G_pred, train_view=train_view)
        
    
