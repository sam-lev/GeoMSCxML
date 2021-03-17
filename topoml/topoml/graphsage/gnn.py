#!/home/sci/samlev/bin/bin/python3                                          
#SBATCH --time=21-00:00:00 # walltime, abbreviated by -t                       
#SBATCH --mem=60G                                                             
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)                                                            
#SBATCH -e slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)                                                            
#SBATCH --gres=gpu:1

from __future__ import division
from __future__ import print_function

import sys
import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

import time

from .models import SampleAndAggregate, SAGEInfo, Node2VecModel
from .minibatch import EdgeMinibatchIterator
from .neigh_samplers import UniformNeighborSampler
from .utils import load_data

from topoml.MSCGNN import MSCGNN
from topoml.ui.LocalSetup import LocalSetup
from topoml.graphsage.utils import random_walk_embedding
from topoml.graphsage.utils import format_data

#supervised trainer imports
from topoml.graphsage.supervised_models import SupervisedGraphsage
from topoml.graphsage.models import SAGEInfo
from topoml.graphsage.minibatch import NodeMinibatchIterator
from topoml.graphsage.neigh_samplers import UniformNeighborSampler
from topoml.graphsage.utils import load_data
from topoml.graphsage.prediction import BipartiteEdgePredLayer

#
# ii. Could update loss given msc behaviours with relation to 'negative' and
#       'positive' arcs-loss dependendt on aggregation of previous to current
#       both negative and positive samples.
# ii. Could add layers to allow convergence faster. Currently model is a
#       two layer multiperceptron
# i.  Could add adaptive hyperparamters e.g. loss drop given some epoch
#       number or exponential/other decay
# iv. Persistence affording subsets of graphs could be to a benefit in
#       explanding training set size or allowing training over set with
#       persistence dependence e.g. birth/death of msc added as hyperparmeter
#       during training. Start simple with high and have training set change by
#       lowering persitence iteratively.
# v.  Construct an aggregator that propogates along arcs depending on
#       persistence. Persistence Weighted Aggregator which is more likely to
#       move along high more persistent arcs. Weighted random walk based on
#       persistence.
# i.  Increase/diversify validation set 
#
# Geometric attributes:
#   geomretric no overlap each pixel part of arc is unique arc.
#   instead of each saddle with 4 incident, instead heres arc heres two nodes
#   at end.
#   extend across lines, cosine similarity of normals
#   laplacian adds dark spot on each line so tangential maxima connect in order to connect to minumum and not loose info for bounded region w/ in max 
#   send classification image
#   train on full image classify full image
#   Look into neighborhood properties based on geomretry
#
# -change weight initialization (currently xavier which is good w/ l2 regularization so would need to cater weight to feature attributes)
# - normalize features around zero instead of mean
# -add layers(?)
# -add identity element for features
# -try to overfit small set
# -play with negative bc loss meant to diverge pos from neg samples in embedding
#  - number neg samples plays large role cross entropy log of class seems to be log of number negative used. 

class unsupervised:
    def __init__(self, aggregator = 'graphsage_mean', env = None, msc_collection=None):
        self.aggregator =  aggregator
        self.graph = None
        self.features = None
        self.id_map = None
        self.node_classes = None
        self.msc_collection = msc_collection
        self.slurm = env if env is None else env == 'slurm'
        self.params_set = False

    def set_parameters(self, G=None, feats=None, id_map=None, walks=None, class_map=None
              , train_prefix='', load_walks=False, number_negative_samples=None
              , number_positive_samples=None, embedding_file_out=''
              , learning_rate=None, depth=2, epochs=200, batch_size=512
              , positive_arcs=[], negative_arcs=[]
              , max_degree=64*3, degree_l1=25, degree_l2=10,degree_l3=0
              , weight_decay=0.001, polarity=6, use_embedding=None
                       , random_context=True, total_features = 1
              , gpu=0, val_model='cvt', model_size = 'small', sigmoid=False, env='multivax'
                            ,out_dim_1 = 256
                            , out_dim_2 = 256):

        if not self.params_set:
            ## variables not actually used but implemented for later development
            self.train_prefix = train_prefix
            self.G = G
            self.feats=feats
            self.id_map=id_map
            self.walks=walks
            self.class_map=class_map
            self.positive_arcs = positive_arcs
            self.negative_arcs=negative_arcs
            self.val_model = val_model

            self.LocalSetup = LocalSetup(env=env)
            self.load_walks = load_walks

            self.use_embedding=use_embedding

            slurm = self.slurm if self.slurm is not None else env == 'slurm'
            if slurm != 'slurm':
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            # Set random seed
            seed = 123
            np.random.seed(seed)
            tf.set_random_seed(seed)

            # Settings
            self.flags = tf.app.flags
            flags = self.flags
            self.FLAGS = self.flags.FLAGS

            self.number_positive_samples=0
            self.number_negative_samples = 4
            if number_negative_samples:
                self.number_negative_samples = number_negative_samples

            self.model_name = embedding_file_out

            if not learning_rate:
                learning_rate = 0.001
            self.learning_rate = learning_rate

            self.epochs = epochs

            self.depth = depth

            # define graph embedding dimensionality
            # dimension 2x used value with concat
            dim = int(474. / 10.)

            concat = False  # mean aggregator only one to perform concat
            self.dim_feature_space = int((dim + 1) / 2) if concat else dim

            #end vomit#####################################




            tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                        "Whether to log device placement.")
            # core params..
            self.flags.DEFINE_string('model', self.aggregator,
                                     'model names. See README for possible values.')  # mean aggregator does not perform concat
            self.flags.DEFINE_float('learning_rate', self.learning_rate, 'initial learning rate.')
            self.flags.DEFINE_integer('drop_1', 2, 'epoch to reduce learning rate first time  by a tenth')
            self.flags.DEFINE_integer('drop_2', 175, 'epoch to reduce learning rate for the second time by a tenth')
            self.flags.DEFINE_string("model_size", model_size, "Can be big or small; model specific def'ns")
            self.flags.DEFINE_string('train_prefix', train_prefix,
                                     'name of the object file that stores the training data. must be specified.')

            self.flags.DEFINE_string('model_name', self.model_name, 'name of the embedded graph model file is created.')

            self.flags.DEFINE_integer('depth', self.depth,
                                      'epoch to reduce learning rate for the second time by a tenth')  # I added this, journal advocates depth of 2 but loss seems to improve with more

            # left to default values in main experiments
            self.flags.DEFINE_integer('epochs', self.epochs, 'number of epochs to train.')
            self.flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
            self.flags.DEFINE_float('weight_decay', weight_decay, 'weight for l2 loss on embedding matrix.')
            self.flags.DEFINE_integer('max_degree', max_degree, 'maximum node degree.')  # 64*3
            self.flags.DEFINE_integer('samples_1', degree_l1,'number of samples in layer 1')  # neighborhood sample size-currently set to whats used in paper, list of samples of variable hops away for convolving at each layer Length #layers +1
            self.flags.DEFINE_integer('samples_hidden', degree_l1, 'number of samples in hidden layers')
            self.flags.DEFINE_integer('samples_2', degree_l2, 'number of users samples in layer 2')  # neighborhoos sample size
            self.flags.DEFINE_integer('dim_1', out_dim_1,
                                      'Size of output dim (final is 2x this, if using concat)')  # mean aggregator does not perform concat else do, list of dimensions of the hidden representations from the input layer to the final latyer, length #Layers+1
            self.flags.DEFINE_integer('dim_hidden', out_dim_1, 'Size of output dim (final is 2x this, if using concat)')
            self.flags.DEFINE_integer('dim_2', out_dim_2, 'Size of output dim (final is 2x this, if using concat)')
            self.flags.DEFINE_boolean('random_context', random_context, 'Whether to use random context or direct edges')
            self.flags.DEFINE_integer('neg_sample_size', polarity, 'number of negative samples')  # paper hard set to twenty rather than actual negative. defines the 'weight' on which neighboring negative nodes have on the loss function allowing a spread in the embedding space of positive and negative samples.
            self.flags.DEFINE_integer('batch_size', batch_size, 'minibatch size.')
            self.flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')  # node to vector
            self.flags.DEFINE_integer('identity_dim', 0,
                                      'Set to positive value to use identity embedding features of that dimension. Default 0.')

            # logging, saving, validation settings etc.
            self.flags.DEFINE_boolean('save_embeddings', True,
                                      'whether to save embeddings for all nodes after training')
            self.flags.DEFINE_string('base_log_dir', './log-dir', 'base directory for logging and saving embeddings')
            self.flags.DEFINE_integer('validate_iter', 1000, "how often to run a validation minibatch.")
            self.flags.DEFINE_integer('validate_batch_size', 5, "how many nodes per validation sample.")
            self.flags.DEFINE_integer('gpu', gpu, "which gpu to use.")
            self.flags.DEFINE_string('env', 'multivax', 'environment to manage data paths and gpu use')
            self.flags.DEFINE_integer('print_every', 650, "How often to print training info.")
            self.flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")


            if slurm != 'slurm':
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.FLAGS.gpu)

            self.GPU_MEM_FRACTION = 0.95

            self.params_set = True

    def train(self, G=None, feats=None, id_map=None, walks=None, class_map=None
              , train_prefix='', load_walks=False, number_negative_samples=None
              , number_positive_samples=None, embedding_file_out=''
              , learning_rate=None, depth=3, epochs=200, batch_size=512
              , positive_arcs=[], negative_arcs=[]
              , weight_decay=0.001, polarity=6, use_embedding=None
              , max_degree=64 * 3, degree_l1=25, degree_l2=10, degree_l3=0
              , gpu=0, env='mutivax', sigmoid=False, val_model='cvt', model_size="small"
                            ,out_dim_1 = 256
                            , out_dim_2 = 256,
              total_features = 1):

        if load_walks is not None or walks is not None:
            random_context = True
        else:
            random_context  = False

        self.set_parameters(G=G, feats=feats, id_map=id_map, walks=walks, class_map=class_map
              , train_prefix=train_prefix, load_walks=load_walks, number_negative_samples=number_negative_samples
              , number_positive_samples=number_positive_samples, embedding_file_out=embedding_file_out
              , learning_rate=learning_rate, depth=depth, epochs=epochs, batch_size=batch_size
              , positive_arcs=positive_arcs, negative_arcs=negative_arcs
              , max_degree=max_degree, degree_l1=degree_l1, degree_l2=degree_l2,degree_l3=degree_l3
              , weight_decay=weight_decay, polarity=polarity, use_embedding=use_embedding
                            , random_context=random_context, total_features = total_features
              , gpu=gpu, val_model=val_model, sigmoid=sigmoid, env=env, model_size=model_size
                            ,out_dim_1 = out_dim_1
                            , out_dim_2 = out_dim_2)

        if False:
            self.LocalSetup = LocalSetup(env=env)
            self.load_walks = load_walks

            slurm = self.slurm if self.slurm is not None else env == 'slurm'
            if slurm != 'slurm':
                os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

            # Set random seed
            seed = 123
            np.random.seed(seed)
            tf.set_random_seed(seed)

            # Settings
            self.flags = tf.app.flags
            self.FLAGS = self.flags.FLAGS

            self.number_negative_samples = 4
            if number_negative_samples:
                self.number_negative_samples = number_negative_samples

            self.model_name = embedding_file_out

            if not learning_rate:
                learning_rate = 0.001
            self.learning_rate = learning_rate

            self.epochs = epochs

            self.depth = depth

            # define graph embedding dimensionality
            # dimension 2x used value with concat
            dim = int(474./10.)

            concat = False #mean aggregator only one to perform concat
            self.dim_feature_space = int((dim+1)/2) if concat else dim

            self.generate_embedding = use_embedding is not None
            self.use_embedding = use_embedding
            self.train_prefix = train_prefix
            self.G = G
            self.feats = feats


            if use_embedding is not None:
                G = use_embedding[0] # graph to embed
                feats = use_embedding[1] # features of graph to embed
                id_map = use_embedding[2] # learned embedding id map
                walks = use_embedding[3] if load_walks is None else []
                class_map = []
            # Collect training data
            # train from saved file, assumes pre-labeled train/test nodes
            if self.msc_collection is None:
                if train_prefix and G is None:
                    print("loading graph data for gnn training")
                    self.train_prefix = train_prefix


                    train_data = load_data(train_prefix, load_walks=load_walks, scheme_required = True, train_or_test='train')

                    self.number_negative_samples = train_data[len(train_data)-2]
                    number_positive_samples = train_data[len(train_data)-1]
                    number_samples = self.number_negative_samples + number_positive_samples
                    proportion_negative = int(number_samples/float(self.number_negative_samples))

                # train from passed graph, assumed pre-labeled(/processed)
                # graph with test/train nodes
                elif G is not None and feats is not None and  id_map is not None and class_map is not None and not train_prefix:
                    train_prefix = 'nNeg-'+str(number_negative_samples)+'nPos-'+str(number_positive_samples)
                    print("using pre-processed graph data for gnn training")
                    self.number_negative_samples = number_negative_samples
                    number_samples= number_negative_samples+number_positive_samples
                    proportion_negative = int(number_samples/float(number_negative_samples))
                    train_data = (G, feats, id_map, walks, class_map, [], [])

                # train from cvt sampled graph and respective in/out arcs as train
                elif positive_arcs and negative_arcs:
                    train_data = load_data(positive_arcs, negative_arcs, load_walks=load_walks, scheme_required = True, train_or_test='train')
                    self.number_negative_samples =  len(negative_arcs)
                    number_samples = len(positive_arcs)+len(negative_arcs)
                    proportion_negative = int(number_samples/float(self.number_negative_samples))
                #keep labeled (test/train) graph for later use in testing
                self.graph = train_data[0]
                self.features = train_data[1]
                self.id_map = train_data[2]
                self.node_classes = train_data[4]

                if load_walks:
                    walks = []
                    if isinstance(self.graph.nodes()[0], int):
                        conversion = lambda n: int(n)
                    else:
                        conversion = lambda n: n
                    with open( load_walks + "-walks.txt") as fp:
                        for line in fp:
                            walks.append(map(conversion, line.split()))

                train_data = (train_data[0], train_data[1], train_data[2], walks, train_data[4], train_data[5], train_data[6])



        train_data = self.get_data()
        #begin training
        print('Begin GNN training')
        print('')
        #if self.msc_collection is None:
        self._train(train_data[:-2])
        #else:
        #    self.batch_train(self.msc_collection)

    def log_dir(self):
        log_dir = self.FLAGS.base_log_dir + "/" + self.FLAGS.model_name+"-unsup-json_graphs" #+ self.FLAGS.train_prefix.split("/")[-2]
        log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
                model=self.FLAGS.model,
                model_size=self.FLAGS.model_size,
                lr=self.FLAGS.learning_rate)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    # Define model evaluation function
    def evaluate(self, sess, model, minibatch_iter, size=None):
        t_test = time.time()
        feed_dict_val = minibatch_iter.val_feed_dict(size)
        outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                            feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

    def incremental_evaluate(self, sess, model, minibatch_iter, size):
        t_test = time.time()
        finished = False
        val_losses = []
        val_mrrs = []
        iter_num = 0
        while not finished:
            feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
            iter_num += 1
            outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                                feed_dict=feed_dict_val)
            val_losses.append(outs_val[0])
            val_mrrs.append(outs_val[2])
        return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)

    def save_val_embeddings(self,sess, model, minibatch_iter, size, out_dir, mod=""):
        val_embeddings = []
        finished = False
        seen = set([])
        nodes = []
        iter_num = 0
        name = "val"
        while not finished:
            feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
            iter_num += 1
            outs_val = sess.run([model.loss, model.mrr, model.outputs1], 
                                feed_dict=feed_dict_val)
            #ONLY SAVE FOR embeds1 because of planetoid
            for i, edge in enumerate(edges):
                if not edge[0] in seen:
                    val_embeddings.append(outs_val[-1][i,:])
                    nodes.append(edge[0])
                    seen.add(edge[0])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        val_embeddings = np.vstack(val_embeddings)
        np.save(out_dir + name + mod + ".npy",  val_embeddings)
        with open(out_dir + name + mod + ".txt", "w") as fp:
            fp.write("\n".join(map(str,nodes)))

    def get_graph(self):
        return self.G

    def construct_placeholders(self):
        # Define placeholders
        placeholders = {
            'batch1' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
            'batch2' : tf.placeholder(tf.int32, shape=(None), name='batch2'),
            # negative samples for all nodes in the batch
            'neg_samples': tf.placeholder(tf.int32, shape=(None,),
                name='neg_sample_size'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
        }
        return placeholders

    def _train(self, train_data, test_data=None):
        G = train_data[0]
        features = train_data[1]
        id_map = train_data[2]
        
        if not features is None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        context_pairs = train_data[3] if self.FLAGS.random_context else None
        placeholders = self.construct_placeholders()
        minibatch = EdgeMinibatchIterator(G, 
                id_map,
                placeholders, batch_size=self.FLAGS.batch_size,
                max_degree=self.FLAGS.max_degree, 
                num_neg_samples=self.FLAGS.neg_sample_size,
                context_pairs = context_pairs)
        adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        if self.FLAGS.model == 'graphsage_mean':
            # Create model
            # for more layers add layers to MLP in models.py as well as
            # add SAGEInfo nodes for more layers [layer name, neighbor sampler,
            #               number neighbors sampled, out dim]
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]
            for l in range(self.depth-2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_hidden, 2*self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append(SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))

            model = SampleAndAggregate(placeholders, 
                                         features,
                                         adj_info,
                                         minibatch.deg,
                                         layer_infos=layer_infos,
                                         depth = self.depth,
                                         model_size=self.FLAGS.model_size,
                                         identity_dim = self.FLAGS.identity_dim,
                                         logging=True)
            """callbacks=[
                ModelSaver(), # Record state graph at intervals during epochs
                InferenceRunner(dataset_train,
                                [ScalarStats('cost'), ClassificationError()]), #Compare to validation set
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)]) # denote current hyperparameters
            ],"""
        elif self.FLAGS.model == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, 2*self.FLAGS.dim_1)]
            for l in range(self.depth-2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_hidden, 2*self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append( SAGEInfo("node", sampler, self.FLAGS.samples_2, 2*self.FLAGS.dim_2))
            #layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, 2*self.FLAGS.dim_1),
            #                    SAGEInfo("node", sampler, self.FLAGS.samples_2, 2*self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders, 
                                         features,
                                         adj_info,
                                         minibatch.deg,
                                         layer_infos=layer_infos, 
                                         aggregator_type="gcn",
                                         model_size=self.FLAGS.model_size,
                                         identity_dim = self.FLAGS.identity_dim,
                                         concat=False,
                                         logging=True)

        elif self.FLAGS.model == 'graphsage_seq':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                                SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders, 
                                         features,
                                         adj_info,
                                         minibatch.deg,
                                         layer_infos=layer_infos, 
                                         identity_dim = self.FLAGS.identity_dim,
                                         aggregator_type="seq",
                                         model_size=self.FLAGS.model_size,
                                         logging=True)

        elif self.FLAGS.model == 'graphsage_maxpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]
            for l in range(self.depth-2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_hidden, 2*self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append(SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            #layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
            #                    SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]
            model = SampleAndAggregate(placeholders, 
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                         layer_infos=layer_infos, 
                                         aggregator_type="maxpool",
                                         model_size=self.FLAGS.model_size,
                                         identity_dim = self.FLAGS.identity_dim,
                                         logging=True)
            """callbacks = [
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.001), (self.FLAGS.drop_1, 0.0001), (self.FLAGS.drop_2, 0.00001)])
                ]"""
        elif self.FLAGS.model == 'graphsage_meanpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]
            for l in range(self.depth-2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_hidden, self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append(SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            #layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
            #                    SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders, 
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                         layer_infos=layer_infos, 
                                         aggregator_type="meanpool",
                                         model_size=self.FLAGS.model_size,
                                         identity_dim = self.FLAGS.identity_dim,
                                         logging=True)

        elif self.FLAGS.model == 'n2v':
            model = Node2VecModel(placeholders, features.shape[0],
                                           minibatch.deg,
                                           #2x because graphsage uses concat
                                           nodevec_dim=2*self.FLAGS.dim_1,
                                           lr=self.FLAGS.learning_rate)
        else:
            raise Exception('Error: model name unrecognized.')

        config = tf.ConfigProto(log_device_placement=self.FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.GPU_MEM_FRACTION
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_dir(), sess.graph)

        # Init variables
        sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

        # Train model

        train_shadow_mrr = None
        shadow_mrr = None

        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        train_adj_info = tf.assign(adj_info, minibatch.adj)
        val_adj_info = tf.assign(adj_info, minibatch.test_adj)

        if self.generate_embedding:
            self.FLAGS.epochs = 1

        for epoch in range(self.FLAGS.epochs): 
            minibatch.shuffle() 

            iter = 0
            print('...')
            print('Epoch: %04d' % (epoch + 1))
            print('...')

            epoch_val_costs.append(0)
            while not minibatch.end():
                # Construct feed dictionary
                feed_dict = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})

                t = time.time()
                # Training step
                outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all, 
                        model.mrr, model.outputs1], feed_dict=feed_dict)
                train_cost = outs[2]
                train_mrr = outs[5]
                if train_shadow_mrr is None:
                    train_shadow_mrr = train_mrr#
                else:
                    train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)

                if iter % self.FLAGS.validate_iter == 0:
                    # Validation
                    sess.run(val_adj_info.op)
                    val_cost, ranks, val_mrr, duration  = self.evaluate(sess, model, minibatch, size=self.FLAGS.validate_batch_size)
                    sess.run(train_adj_info.op)
                    epoch_val_costs[-1] += val_cost
                if shadow_mrr is None:
                    shadow_mrr = val_mrr
                else:
                    shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

                if total_steps % self.FLAGS.print_every == 0:
                    summary_writer.add_summary(outs[0], total_steps)

                # Print results
                avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

                if total_steps % self.FLAGS.print_every == 0:
                    print("Iter:", '%04d' % iter, 
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_mrr=", "{:.5f}".format(train_mrr), 
                          "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr), # exponential moving average
                          "val_loss=", "{:.5f}".format(val_cost),
                          "val_mrr=", "{:.5f}".format(val_mrr), 
                          "val_mrr_ema=", "{:.5f}".format(shadow_mrr), # exponential moving average
                          "time=", "{:.5f}".format(avg_time))

                iter += 1
                total_steps += 1

                if total_steps > self.FLAGS.max_total_steps:
                    break

            if total_steps > self.FLAGS.max_total_steps:
                    break

        print("Optimization Finished.")
        # Modify for embedding of new graph
        #   adj_info for unseen needed ( node connectivity to rest of new graph)
        #   modify minibatch to accomodate new adj_info of unseen
        #   also feature matrix for unseen graph
        if self.FLAGS.save_embeddings:
            sess.run(val_adj_info.op)

            self.save_val_embeddings(sess, model, minibatch, self.FLAGS.validate_batch_size, self.log_dir())

            if self.FLAGS.model == "n2v":
                # stopping the gradient for the already trained nodes
                train_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
                        dtype=tf.int32)
                test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']], 
                        dtype=tf.int32)
                update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
                no_update_nodes = tf.nn.embedding_lookup(model.context_embeds,tf.squeeze(train_ids))
                update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
                no_update_nodes = tf.stop_gradient(tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
                model.context_embeds = update_nodes + no_update_nodes
                sess.run(model.context_embeds)

                # run random walks
                from .utils import run_random_walks
                nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
                start_time = time.time()
                pairs = run_random_walks(G, nodes, num_walks=50)
                walk_time = time.time() - start_time

                test_minibatch = EdgeMinibatchIterator(G, 
                    id_map,
                    placeholders, batch_size=self.FLAGS.batch_size,
                    max_degree=self.FLAGS.max_degree, 
                    num_neg_samples=self.FLAGS.neg_sample_size,
                    context_pairs = pairs,
                    n2v_retrain=True,
                    fixed_n2v=True)

                start_time = time.time()
                print("Doing test training for n2v.")
                test_steps = 0
                for epoch in range(self.FLAGS.n2v_test_epochs):
                    test_minibatch.shuffle()
                    while not test_minibatch.end():
                        feed_dict = test_minibatch.next_minibatch_feed_dict()
                        feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})
                        outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all, 
                            model.mrr, model.outputs1], feed_dict=feed_dict)
                        if test_steps % self.FLAGS.print_every == 0:
                            print("Iter:", '%04d' % test_steps, 
                                  "train_loss=", "{:.5f}".format(outs[1]),
                                  "train_mrr=", "{:.5f}".format(outs[-2]))
                        test_steps += 1
                train_time = time.time() - start_time
                self.save_val_embeddings(sess, model, minibatch, self.FLAGS.validate_batch_size, self.log_dir(), mod="-test")
                print("Total time: ", train_time+walk_time)
                print("Walk time: ", walk_time)
                print("Train time: ", train_time)

    def get_data(self, train_data=None):
        print("Loading training data..")
        #train_data = load_data(FLAGS.train_prefix)

        ### load MSC data
        print('loading msc graph data')

        self.generate_embedding = self.use_embedding is not None
        if self.use_embedding is not None:
            self.G = self.use_embedding[0]  # graph to embed
            self.feats = self.use_embedding[1]  # features of graph to embed
            self.id_map = self.use_embedding[2]  # learned embedding id map
            self.walks = self.use_embedding[3] if self.load_walks is None else []
            self.class_map = []
        # Collect training data
        # train from saved file, assumes pre-labeled train/test nodes
        if self.msc_collection is None:
            if self.train_prefix and self.G is None:
                print("loading graph data for gnn training")
                self.train_prefix = self.train_prefix

                train_data = load_data(self.train_prefix, load_walks=self.load_walks, scheme_required=True,
                                       train_or_test='train')

                self.number_negative_samples = train_data[len(train_data) - 2]
                self.number_positive_samples = train_data[len(train_data) - 1]
                number_samples = self.number_negative_samples + self.number_positive_samples
                proportion_negative = int(number_samples / float(self.number_negative_samples))

            # train from passed graph, assumed pre-labeled(/processed)
            # graph with test/train nodes
            elif self.G is not None and self.feats is not None and self.id_map is not None and self.class_map is not None and not self.train_prefix:
                train_prefix = 'nNeg-' + str(self.number_negative_samples) + 'nPos-' + str(self.number_positive_samples)
                print("using pre-processed graph data for gnn training")
                self.number_negative_samples = self.number_negative_samples
                number_samples = self.number_negative_samples + self.number_positive_samples
                proportion_negative = int(number_samples / float(self.number_negative_samples))
                train_data = (self.G, self.feats, self.id_map, self.walks, self.class_map, [], [])

            # train from cvt sampled graph and respective in/out arcs as train
            elif self.positive_arcs and self.negative_arcs:
                train_data = load_data(self.positive_arcs, self.negative_arcs, load_walks=self.load_walks, scheme_required=True,
                                       train_or_test='train')
                self.number_negative_samples = len(self.negative_arcs)
                number_samples = len(self.positive_arcs) + len(self.negative_arcs)
                proportion_negative = int(number_samples / float(self.number_negative_samples))
            # keep labeled (test/train) graph for later use in testing
            self.graph = train_data[0]
            self.features = train_data[1]
            self.id_map = train_data[2]
            self.node_classes = train_data[4]

            if self.load_walks:
                walks = []
                if isinstance(self.graph.nodes()[0], int):
                    conversion = lambda n: int(n)
                else:
                    conversion = lambda n: n
                with open(self.load_walks + "-walks.txt") as fp:
                    for line in fp:
                        walks.append(map(conversion, line.split()))

            train_data = (
            train_data[0], train_data[1], train_data[2], walks, train_data[4], train_data[5], train_data[6])
        return train_data

    def format_msc_feature_graph(self, image, msc, mask, segmentation, persistence, blur):

        mscgnn = MSCGNN(msc=msc)
        # add number id to name
        msc_graph_name = 'msc-feature-graph-' + str(persistence) + 'blur-' + str(blur)
        mscgnn.msc_feature_graph(image=np.transpose(np.mean(image, axis=1), (1, 0)), X=image.shape[0], Y=image.shape[2]
                                 , validation_samples=2, validation_hops=20
                                 , test_samples=0, test_hops=0, accuracy_threshold=0.2
                                 , write_json_graph_path='./data', name=msc_graph_name
                                 , test_graph=False)

        if self.load_walks:
            print('... Generating Random Walk Neighborhoods for Node Co-Occurance')
            walk_embedding_file = os.path.join(self.LocalSetup.project_base_path, 'datasets', 'walk_embeddings'
                                               , str(persistence) + str(blur) + 'test_walk')
            random_walk_embedding(mscgnn.G, walk_length=10, number_walks=20, out_file=walk_embedding_file)

        G, feats, id_map \
            , walks, class_map \
            , number_negative_samples \
            , number_positive_samples = format_data(dual=self.G
                                                    , features=self.features
                                                    , node_id=self.node_id
                                                    , id_map=self.node_id
                                                    , node_classes=self.node_classes
                                                    , train_or_test=''
                                                    , scheme_required=True
                                                    , load_walks=self.load_walks)
        training_sample = (G, feats, id_map, walks
                           , class_map, number_negative_samples, number_positive_samples)
        return training_sample

    def batch_train(self, msc_collection, test_data=None):
        mscbatch = []

        training_msc_dataset = msc_collection[0]
        persistence_values = msc_collection[1]
        blur_sigmas = msc_collection[2]
        for image, msc, mask, segmentation in training_msc_dataset:
            # = training_msc_dataset[0]
            pers = 0 #temp
            blur = 0
            msc = msc[(persistence_values[pers], blur_sigmas[blur])]
            training_sample = self.format_msc_feature_graph(image, msc, mask, segmentation, persistence_values[pers], blur_sigmas[blur])
            G = training_sample[0]
            features = training_sample[1]
            id_map = training_sample[2]

            if not features is None:
                # pad with dummy zero vector
                features = np.vstack([features, np.zeros((features.shape[1],))])

            context_pairs = training_sample[3] if self.FLAGS.random_context else None
            placeholders = self.construct_placeholders()
            minibatch = EdgeMinibatchIterator(G,
                                              id_map,
                                              placeholders, batch_size=self.FLAGS.batch_size,
                                              max_degree=self.FLAGS.max_degree,
                                              num_neg_samples=self.FLAGS.neg_sample_size,
                                              context_pairs=context_pairs)
            adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
            adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        if self.FLAGS.model == 'graphsage_mean':
            # Create model
            # for more layers add layers to MLP in models.py as well as
            # add SAGEInfo nodes for more layers [layer name, neighbor sampler,
            #               number neighbors sampled, out dim]
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       depth=self.depth,
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
            """callbacks=[
                ModelSaver(), # Record state graph at intervals during epochs
                InferenceRunner(dataset_train,
                                [ScalarStats('cost'), ClassificationError()]), #Compare to validation set
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)]) # denote current hyperparameters
            ],"""
        elif self.FLAGS.model == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, 2 * self.FLAGS.dim_1)]
            for l in range(self.depth - 2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_hidden, 2 * self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append(SAGEInfo("node", sampler, self.FLAGS.samples_2, 2 * self.FLAGS.dim_2))
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, 2 * self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, 2 * self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       aggregator_type="gcn",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       concat=False,
                                       logging=True)

        elif self.FLAGS.model == 'graphsage_seq':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       identity_dim=self.FLAGS.identity_dim,
                                       aggregator_type="seq",
                                       model_size=self.FLAGS.model_size,
                                       logging=True)

        elif self.FLAGS.model == 'graphsage_maxpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]
            for l in range(self.depth - 2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_hidden, 2 * self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append(SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            # layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
            #                    SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]
            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       aggregator_type="maxpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
            """callbacks = [
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.001), (self.FLAGS.drop_1, 0.0001), (self.FLAGS.drop_2, 0.00001)])
                ]"""
        elif self.FLAGS.model == 'graphsage_meanpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]
            for l in range(self.depth - 2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_hidden, self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append(SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            # layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
            #                    SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       aggregator_type="meanpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)

        elif self.FLAGS.model == 'n2v':
            model = Node2VecModel(placeholders, features.shape[0],
                                  minibatch.deg,
                                  # 2x because graphsage uses concat
                                  nodevec_dim=2 * self.FLAGS.dim_1,
                                  lr=self.FLAGS.learning_rate)
        else:
            raise Exception('Error: model name unrecognized.')

        config = tf.ConfigProto(log_device_placement=self.FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.GPU_MEM_FRACTION
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_dir(), sess.graph)

        # Init variables
        sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

        # Train model

        train_shadow_mrr = None
        shadow_mrr = None

        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        train_adj_info = tf.assign(adj_info, minibatch.adj)
        val_adj_info = tf.assign(adj_info, minibatch.test_adj)

        if self.generate_embedding:
            self.FLAGS.epochs = 1

        for epoch in range(self.FLAGS.epochs):
            minibatch.shuffle()

            iter = 0
            print('...')
            print('Epoch: %04d' % (epoch + 1))
            print('...')

            epoch_val_costs.append(0)
            while not minibatch.end():
                # Construct feed dictionary
                feed_dict = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})

                t = time.time()
                # Training step
                outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all,
                                 model.mrr, model.outputs1], feed_dict=feed_dict)
                train_cost = outs[2]
                train_mrr = outs[5]
                if train_shadow_mrr is None:
                    train_shadow_mrr = train_mrr  #
                else:
                    train_shadow_mrr -= (1 - 0.99) * (train_shadow_mrr - train_mrr)

                if iter % self.FLAGS.validate_iter == 0:
                    # Validation
                    sess.run(val_adj_info.op)
                    val_cost, ranks, val_mrr, duration = self.evaluate(sess, model, minibatch,
                                                                       size=self.FLAGS.validate_batch_size)
                    sess.run(train_adj_info.op)
                    epoch_val_costs[-1] += val_cost
                if shadow_mrr is None:
                    shadow_mrr = val_mrr
                else:
                    shadow_mrr -= (1 - 0.99) * (shadow_mrr - val_mrr)

                if total_steps % self.FLAGS.print_every == 0:
                    summary_writer.add_summary(outs[0], total_steps)

                # Print results
                avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

                if total_steps % self.FLAGS.print_every == 0:
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_mrr=", "{:.5f}".format(train_mrr),
                          "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr),  # exponential moving average
                          "val_loss=", "{:.5f}".format(val_cost),
                          "val_mrr=", "{:.5f}".format(val_mrr),
                          "val_mrr_ema=", "{:.5f}".format(shadow_mrr),  # exponential moving average
                          "time=", "{:.5f}".format(avg_time))

                iter += 1
                total_steps += 1

                if total_steps > self.FLAGS.max_total_steps:
                    break

            if total_steps > self.FLAGS.max_total_steps:
                break

        print("Optimization Finished.")
        if self.FLAGS.save_embeddings:
            sess.run(val_adj_info.op)

            self.save_val_embeddings(sess, model, minibatch, self.FLAGS.validate_batch_size, self.log_dir())

            if self.FLAGS.model == "n2v":
                # stopping the gradient for the already trained nodes
                train_ids = tf.constant(
                    [[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
                    dtype=tf.int32)
                test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']],
                                       dtype=tf.int32)
                update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
                no_update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(train_ids))
                update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
                no_update_nodes = tf.stop_gradient(
                    tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
                model.context_embeds = update_nodes + no_update_nodes
                sess.run(model.context_embeds)

                # run random walks
                from .utils import run_random_walks
                nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
                start_time = time.time()
                pairs = run_random_walks(G, nodes, num_walks=50)
                walk_time = time.time() - start_time

                test_minibatch = EdgeMinibatchIterator(G,
                                                       id_map,
                                                       placeholders, batch_size=self.FLAGS.batch_size,
                                                       max_degree=self.FLAGS.max_degree,
                                                       num_neg_samples=self.FLAGS.neg_sample_size,
                                                       context_pairs=pairs,
                                                       n2v_retrain=True,
                                                       fixed_n2v=True)

                start_time = time.time()
                print("Doing test training for n2v.")
                test_steps = 0
                for epoch in range(self.FLAGS.n2v_test_epochs):
                    test_minibatch.shuffle()
                    while not test_minibatch.end():
                        feed_dict = test_minibatch.next_minibatch_feed_dict()
                        feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})
                        outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all,
                                         model.mrr, model.outputs1], feed_dict=feed_dict)
                        if test_steps % self.FLAGS.print_every == 0:
                            print("Iter:", '%04d' % test_steps,
                                  "train_loss=", "{:.5f}".format(outs[1]),
                                  "train_mrr=", "{:.5f}".format(outs[-2]))
                        test_steps += 1
                train_time = time.time() - start_time
                self.save_val_embeddings(sess, model, minibatch, self.FLAGS.validate_batch_size, self.log_dir(),
                                         mod="-test")
                print("Total time: ", train_time + walk_time)
                print("Walk time: ", walk_time)
                print("Train time: ", train_time)

    if __name__ == '__main__':
        tf.app.run()


class supervised:
    # begin vomit
    def __init__(self, aggregator='graphsage_mean', env=None
                 , msc_collection=None, persistence_values=None, model_path=None):
        self.aggregator = aggregator
        self.graph = None
        self.features = None
        self.id_map = None
        self.node_classes = None
        self.msc_collection = msc_collection
        self.persistence_values=persistence_values
        self.slurm = env if env is None else env == 'slurm'
        self.infer = False
        self.model = None
        self.sess = None
        self.model_path = model_path
        self.FLAGS = None
        self.params_set = False
        self.placeholders = None

    def set_parameters(self, G=None, feats=None, id_map=None, walks=None, class_map=None
              , train_prefix='', load_walks=False, number_negative_samples=None
              , number_positive_samples=None, embedding_file_out=''
              , learning_rate=None, depth=2, epochs=200, batch_size=512
              , positive_arcs=[], negative_arcs=[]
                       ,dim_1=128, dim_2=128
              , max_degree=64*3, degree_l1=25, degree_l2=10,degree_l3=0
              , weight_decay=0.001, polarity=6, use_embedding=None
                       , jumping_knowledge=True, concat=True,
                       jump_type = 'pool'
              , gpu=0, val_model='cvt', model_size="small", sigmoid=False, env='multivax'):

        if True:#not self.params_set:
            ## variables not actually used but implemented for later development
            self.train_prefix = train_prefix
            self.G = G
            self.feats=feats
            self.id_map=id_map
            self.walks=walks
            self.class_map=class_map
            self.positive_arcs = positive_arcs
            self.negative_arcs=negative_arcs
            self.val_model = val_model

            self.LocalSetup = LocalSetup(env=env)
            self.load_walks = load_walks

            self.use_embedding=use_embedding

            slurm = self.slurm if self.slurm is not None else env == 'slurm'
            if slurm != 'slurm':
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            # Set random seed
            seed = 123
            np.random.seed(seed)
            tf.set_random_seed(seed)

            # Settings
            self.flags = tf.app.flags
            flags = self.flags
            self.FLAGS = self.flags.FLAGS

            self.number_positive_samples=0
            self.number_negative_samples = 4
            if number_negative_samples:
                self.number_negative_samples = number_negative_samples

            self.model_name = embedding_file_out

            if not learning_rate:
                learning_rate = 0.001
            self.learning_rate = learning_rate

            self.epochs = epochs

            self.depth = depth

            # define graph embedding dimensionality
            # dimension 2x used value with concat
            dim = int(474. / 10.)

            self.concat = concat  # mean aggregator only one to perform concat
            self.dim_feature_space = int((dim + 1) / 2) if concat else dim
            self.jump_type = jump_type
            #end vomit#####################################


            # Set random seed
            seed = 123
            np.random.seed(seed)
            tf.set_random_seed(seed)

            def del_all_flags(FLAGS):
                flags_dict = FLAGS._flags()
                keys_list = [keys for keys in flags_dict]
                for keys in keys_list:
                    FLAGS.__delattr__(keys)

            del_all_flags(tf.flags.FLAGS)

            tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                        """Whether to log device placement.""")
            # core params..
            flags.DEFINE_string('model', self.aggregator, 'model names. See README for possible values.')
            flags.DEFINE_float('learning_rate', self.learning_rate, 'initial learning rate.')
            flags.DEFINE_string("model_size", model_size, "Can be big or small; model specific def'ns")
            flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

            # left to default values in main experiments
            flags.DEFINE_integer('epochs', self.epochs, 'number of epochs to train.')
            flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
            flags.DEFINE_float('weight_decay', weight_decay, 'weight for l2 loss on embedding matrix.')
            flags.DEFINE_integer('max_degree', max_degree, 'maximum node degree.')#128
            flags.DEFINE_integer('samples_1', degree_l1, 'number of samples in layer 1')#25, number samples per node
            flags.DEFINE_integer('samples_2', degree_l2, 'number of samples in layer 2')#10
            flags.DEFINE_integer('samples_3', degree_l3, 'number of users samples in layer 3. (Only for mean model)')#0
            flags.DEFINE_integer('dim_1', dim_1, 'Size of output dim (final is 2x this, if using concat)')
            flags.DEFINE_integer('dim_2', dim_2, 'Size of output dim (final is 2x this, if using concat)')
            flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')#true
            flags.DEFINE_integer('batch_size', batch_size, 'minibatch size.')
            flags.DEFINE_boolean('sigmoid', sigmoid, 'whether to use sigmoid loss')
            flags.DEFINE_integer('identity_dim', 0,
                                 'Set to positive value to use identity embedding features of that dimension. Default 0.')

            self.flags.DEFINE_boolean('jumping_knowledge', jumping_knowledge, 'whether to use jumping knowledge approach for graph embedding')
            # logging, saving, validation settings etc.
            self.flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
            self.flags.DEFINE_string('base_log_dir', './log-dir', 'base directory for logging and saving embeddings')
            self.flags.DEFINE_integer('validate_iter', 100, "how often to run a validation minibatch.")
            self.flags.DEFINE_integer('validate_batch_size', batch_size//4, "how many nodes per validation sample.")
            self.flags.DEFINE_integer('gpu', gpu, "which gpu to use.")
            self.flags.DEFINE_string('env', 'multivax', 'environment to manage data paths and gpu use')
            self.flags.DEFINE_integer('print_every', 50, "How often to print training info.")
            self.flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")
            self.flags.DEFINE_string('model_name', self.model_name, 'name of the embedded graph model file is created.')
            self.flags.DEFINE_integer('depth', self.depth,
                                      'epoch to reduce learning rate for the second time by a tenth')  # I added this, journal advocates depth of 2 but loss seems to improve with more

            if slurm != 'slurm':
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.FLAGS.gpu)

            self.GPU_MEM_FRACTION = 0.98

            #self.params_set = True

    def train(self, G=None, feats=None, id_map=None, walks=None, class_map=None
              , train_prefix='', load_walks=False, number_negative_samples=None
              , number_positive_samples=None, embedding_file_out=''
              , learning_rate=None, depth=3, epochs=200, batch_size=512
              , positive_arcs=[], negative_arcs=[]
              , max_degree=64*3, degree_l1=25, degree_l2=10,degree_l3=0
              , dim_1 = 256, dim_2 = 256,
              concat = True,
              jumping_knowledge = False,
              jump_type = 'pool'
              , weight_decay=0.001, polarity=6, use_embedding=None
              , gpu=0, val_model='cvt', sigmoid=False,model_size="small", env='multivax'):

        self.set_parameters(G=G, feats=feats, id_map=id_map, walks=walks, class_map=class_map
              , train_prefix=train_prefix, load_walks=load_walks, number_negative_samples=number_negative_samples
              , number_positive_samples=number_positive_samples, embedding_file_out=embedding_file_out
              , learning_rate=learning_rate, depth=depth, epochs=epochs, batch_size=batch_size
              , positive_arcs=positive_arcs, negative_arcs=negative_arcs
              , max_degree=max_degree, degree_l1=degree_l1, degree_l2=degree_l2,degree_l3=degree_l3
                            ,dim_1=dim_1,dim_2=dim_2
                            , jumping_knowledge=jumping_knowledge, concat=concat,
                            jump_type = jump_type
              , weight_decay=weight_decay, polarity=polarity, use_embedding=use_embedding
              , gpu=gpu, val_model=val_model, model_size=model_size, sigmoid=sigmoid, env=env)


        # format or retriev data
        train_data = self.get_data()
        ## use msc graph data
        # begin training
        print('Begin GNN training')
        print('')

        # resulting trained model
        self.model = None
        #if self.msc_collection is None and not self.infer:
        print("%%%%%BEGINING TRAINGING%%%%")
        self._train(train_data[:-2])

        # need to union graphs for batch training, either here or prior
        # i.e. can't iterate over disjoint due to poor implementation
        #elif not self.infer:
        #    print("%%%%%BEGINNING BATCH TRAIN%%%%")
        #    self.batch_train(self.msc_collection)


    def calc_f1(self,  y_true, y_pred):
        if not self.FLAGS.sigmoid:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
        return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

    def pred_values(self, pred):
        if not self.FLAGS.sigmoid:
            pred = np.argmax(pred, axis=1)
        else:
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
        return pred

    # Define model evaluation function
    def evaluate(self, sess, model, minibatch_iter, size=None):
        val_preds = []
        t_test = time.time()
        feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)

        # add inference labels
        minibatch_iter.update_batch_prediction(node_outs_val[0])

        val_preds = self.pred_values(node_outs_val[0])
        #minibatch_iter.update_batch_prediction(val_preds)

        mic, mac = self.calc_f1(labels, node_outs_val[0])
        return node_outs_val[1], mic, mac, (time.time() - t_test)

    def log_dir(self):
        log_dir = self.FLAGS.base_log_dir + "/sup-" + self.FLAGS.model_name
        #log_dir = self.FLAGS.base_log_dir + "/sup-" + self.FLAGS.train_prefix.split("/")[-2]
        log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=self.FLAGS.model,
            model_size=self.FLAGS.model_size,
            lr=self.FLAGS.learning_rate)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def incremental_evaluate(self, sess, model, minibatch_iter, size, test=False, inference=False,
                             infer_feed_dict=None, infer_labels = None):
        t_test = time.time()
        val_losses = []
        val_preds = []
        labels = []
        iter_num = 0
        finished = False
        while not finished:
            if infer_feed_dict is None:
                feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(size, iter_num,
                                                                  test=test, inference=inference)
                if inference:
                    finished = True
            else:
                #feed_dict_val, batch_inf, batch_labels = infer_feed_dict#minibatch_iter.inference_feed_dict()
                feed_dict_val = infer_feed_dict
                batch_labels = infer_labels
                finished = True
            node_outs_val = sess.run([model.preds, model.loss],
                                     feed_dict=feed_dict_val)
            val_preds.append(node_outs_val[0])

            # add inference labels
            minibatch_iter.update_batch_prediction(node_outs_val[0])

            preds = self.pred_values(node_outs_val[0])

            #minibatch_iter.update_batch_prediction(val_preds)

            labels.append(batch_labels)
            val_losses.append(node_outs_val[1])
            iter_num += 1
        val_preds = np.vstack(val_preds)
        labels = np.vstack(labels)
        f1_scores = self.calc_f1(labels, val_preds)
        return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)

    def _make_label_vec(self, node, class_map, num_classes):
        label = class_map[node]
        if isinstance(label, list):
            label_vec = np.array(label, dtype=np.float32)
        else:
            label_vec = np.zeros((num_classes), dtype=np.float32)
            class_ind = class_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_placeholders(self, num_classes, batch=None, labels = None, inf_batch_shape=None,
                               class_map=None, id_map = None, name_prefix='', infer=False):
        # Define placeholders
        inf_batch_shape=None
        if False:#inf_batch_shape is not None:
            print("setting", inf_batch_shape)
            bs = tf.placeholder_with_default(inf_batch_shape, shape=(), name=name_prefix +'batch_size')
            #labels = np.vstack([self._make_label_vec(node,class_map,num_classes) for node in batch])
            print("labels shape", labels.shape)
            ls = tf.placeholder_with_default(labels,shape=(inf_batch_shape, num_classes)
                                             , name=name_prefix+'labels')
            #ls2= tf.placeholder_with_default(labels, shape=(inf_batch_shape, num_classes)
            #                                 , name=name_prefix +'_infer_'+'labels2')
            batch_from_map = [id_map[n] for n in batch]
            b = tf.placeholder_with_default(batch_from_map, shape=(inf_batch_shape)
                                            , name=name_prefix+'batch')
            dpo = tf.placeholder_with_default(0., shape=(), name=name_prefix+'dropout')
            #b1 = tf.placeholder_with_default(batch_from_map, shape=(inf_batch_shape)
            #                                 , name=name_prefix + '_infer_'+'batch1')
            training_bullshit_ph = tf.get_default_graph().get_tensor_by_name('batch_size')
            assert bs == training_bullshit_ph
            old_labels= tf.get_default_graph().get_tensor_by_name('labels')
            assert ls ==old_labels
            old_batch = tf.get_default_graph().get_tensor_by_name('batch')
            assert b == old_batch
            old_dropout = tf.get_default_graph().get_tensor_by_name('dropout')
            assert dpo == old_dropout

        else:
            bs = tf.placeholder(tf.int32, shape=(), name="batch_size")
            ls = tf.placeholder(tf.float32, shape=(inf_batch_shape, num_classes), name="labels")
            #ls2 = tf.placeholder(tf.float32, shape=(inf_batch_shape, num_classes), name=name_prefix + 'labels2')
            b = tf.placeholder(tf.int32, shape=(inf_batch_shape), name="batch")
            dpo = tf.placeholder_with_default(0., shape=(),  name="dropout")
            #b1 = tf.placeholder(tf.int32, shape=(inf_batch_shape), name=name_prefix + 'batch1')
        if not infer:
            placeholders = {
                'labels': ls,
                'batch': b,
                'dropout': dpo,
                'batch_size': bs, #tf.placeholder(tf.int32,inf_batch_shape, name=name_prefix+'batch_size'),
            }
        else:
            placeholders = {
                'batch' : b,
                'labels' : ls,
                #'batch1' : b1,                         #tf.placeholder(tf.int32, shape=(None), name='batch1'),
                #'batch2' : ls2,                        #tf.placeholder(tf.int32, shape=(None), name='batch2'),
                # negative samples for all nodes in the batch
                #'neg_samples': tf.placeholder(tf.int32, shape=(None,),
                #    name='neg_sample_size'),
                'dropout': dpo,
                'batch_size' : bs,                    #tf.placeholder(tf.int32, name='batch_size'),
            }
        return placeholders

    def fetch_placeholders(self, batch, labels):


        #bs = tf.get_default_graph().get_tensor_by_name("batch_size")
        #ls = tf.get_default_graph().get_tensor_by_name("labels")
        #b = tf.get_default_graph().get_tensor_by_name("batch")
        #dpo = tf.get_default_graph().get_tensor_by_name("dropout")
        placeholders = {
            'batch': batch,#b,
            'labels': labels, #ls,
            # 'batch1' : b1,                         #tf.placeholder(tf.int32, shape=(None), name='batch1'),
            # 'batch2' : ls2,                        #tf.placeholder(tf.int32, shape=(None), name='batch2'),
            # negative samples for all nodes in the batch
            # 'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            #    name='neg_sample_size'),
            'dropout': 0.,#dpo,
            'batch_size':  len(batch), #bs,  # tf.placeholder(tf.int32, name='batch_size'),
        }
        return placeholders

    def _train(self, train_data, test_data=None):


        start_time = time.time()

        G = train_data[0]
        features = train_data[1]
        id_map = train_data[2]
        class_map = train_data[4]
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
        else:
            num_classes = len(set(class_map.values()))

        if not features is None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        FLAGS = self.FLAGS

        context_pairs = train_data[3] if FLAGS.random_context else None
        placeholders = self.construct_placeholders(num_classes)
        self.placeholders = placeholders

        minibatch = NodeMinibatchIterator(G,
                                          id_map,
                                          placeholders,
                                          class_map,
                                          num_classes,
                                          batch_size=FLAGS.batch_size,
                                          max_degree=FLAGS.max_degree,
                                          context_pairs=context_pairs)
        self.adj_info_ph = tf.placeholder(tf.int32, shape=(None, None))#minibatch.adj.shape)
        adj_info_ph = self.adj_info_ph
        adj_info = tf.Variable(initial_value=minibatch.adj, shape=minibatch.adj.shape
                               , trainable=False, name="adj_info", dtype=tf.int32)
        #adj_info = tf.Variable(adj_info_ph, shape=minibatch.adj.shape, trainable=False, name="adj_info")
        self.adj_info = adj_info

        print(" ... Using aggregator: ", FLAGS.model)

        if FLAGS.model == 'graphsage_mean':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            if FLAGS.samples_3 != 0:
                layer_infos = [SAGEInfo("node_gsmean", sampler, FLAGS.samples_1, FLAGS.dim_1),
                               SAGEInfo("node_gsmean", sampler, FLAGS.samples_2, FLAGS.dim_2),
                               SAGEInfo("node_gsmean", sampler, FLAGS.samples_3, FLAGS.dim_2)]
            elif FLAGS.samples_2 != 0:
                layer_infos = [SAGEInfo("node_gsmean", sampler, FLAGS.samples_1, FLAGS.dim_1),
                               SAGEInfo("node_gsmean", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            else:
                layer_infos = [SAGEInfo("node_gsmean", sampler, FLAGS.samples_1, FLAGS.dim_1)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos,
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)
        elif FLAGS.model == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node_gcn", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                           SAGEInfo("node_gcn", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        aggregator_type="gcn",
                                        model_size=FLAGS.model_size,
                                        concat=False,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)

        elif FLAGS.model == 'graphsage_seq':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node_gsseq", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node_gsseq", sampler, FLAGS.samples_2, FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        aggregator_type="seq",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)

        elif FLAGS.model == 'graphsage_maxpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth+1):
                layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        concat=self.concat,
                                        jumping_knowledge=FLAGS.jumping_knowledge,
                                        jump_type = self.jump_type,
                                        layer_infos=layer_infos,
                                        aggregator_type="maxpool",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)

        elif FLAGS.model == 'graphsage_meanpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node_meanpool", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node_meanpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        aggregator_type="meanpool",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)

        else:
            raise Exception('Error: model name unrecognized.')

        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.GPU_MEM_FRACTION
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()
        #summary_writer = tf.summary.FileWriter(self.log_dir(), sess.graph)

        # Init variables
        sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

        print("Training adj graph shape: ", minibatch.adj.shape)
        print("Test trian adj shape: ", minibatch.test_adj.shape)
        # save during training
        #saver = tf.compat.v1.train.Saver(tf.all_variables()) #save_relative_paths=True)
        model_checkpoint = self.log_dir()+'model-session.ckpt'
        # Train model

        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        train_adj_info = tf.assign(adj_info, minibatch.adj)
        val_adj_info = tf.assign(adj_info, minibatch.test_adj,name='adj_assign')
        for epoch in range(FLAGS.epochs):
            minibatch.shuffle()

            iter = 0
            print('Epoch: %04d' % (epoch + 1))
            epoch_val_costs.append(0)
            while not minibatch.end():
                # Construct feed dictionary
                feed_dict, labels = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                t = time.time()
                # Training step
                outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
                train_cost = outs[2]

                preds = self.pred_values(outs[-1])
                minibatch.update_batch_prediction(preds)

                if iter % FLAGS.validate_iter == 0:
                    # Validation
                    sess.run(val_adj_info.op)
                    if FLAGS.validate_batch_size == -1:
                        val_cost, val_f1_mic, val_f1_mac, duration = self.incremental_evaluate(sess, model, minibatch,
                                                                                          FLAGS.batch_size)
                    else:
                        val_cost, val_f1_mic, val_f1_mac, duration = self.evaluate(sess, model, minibatch,
                                                                              FLAGS.validate_batch_size)
                    sess.run(train_adj_info.op)
                    epoch_val_costs[-1] += val_cost

                    #saver.save(sess, model_checkpoint)#, global_step=total_steps)

                #if total_steps % FLAGS.print_every == 0:
                #    summary_writer.add_summary(outs[0], total_steps)

                # Print results
                avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

                if total_steps % FLAGS.print_every == 0:
                    train_f1_mic, train_f1_mac = self.calc_f1(labels, outs[-1])
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                          "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                          "val_loss=", "{:.5f}".format(val_cost),
                          "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                          "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                          "time=", "{:.5f}".format(avg_time))

                iter += 1
                total_steps += 1

                if total_steps > FLAGS.max_total_steps:
                    break

            if total_steps > FLAGS.max_total_steps:
                break

        print("Optimization Finished!")
        sess.run(val_adj_info.op)
        val_cost, val_f1_mic, val_f1_mac, duration = self.incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
        print("Full validation stats:",
              "loss=", "{:.5f}".format(val_cost),
              "f1_micro=", "{:.5f}".format(val_f1_mic),
              "f1_macro=", "{:.5f}".format(val_f1_mac),
              "time=", "{:.5f}".format(duration))
        with open(self.log_dir() + "val_stats.txt", "w") as fp:
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
                     format(val_cost, val_f1_mic, val_f1_mac, duration))

        end_train_time = time.time()

        print('..Time taken to train model: ', end_train_time-start_time)


        val_cost, val_f1_mic, val_f1_mac, duration = self.incremental_evaluate(sess, model, minibatch, FLAGS.batch_size,
                                                                          test=True)
        print('..Time taken to perform inference: ', end_train_time-time.time())

        print("Writing test set stats to file (don't peak!)")
        with open(self.log_dir() + "test_stats.txt", "w") as fp:
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                     format(val_cost, val_f1_mic, val_f1_mac))

        # save session
        print(" >>>>> Saving final session in: ", self.log_dir())
        #saver.save(sess, model_checkpoint)#, global_step=FLAGS.max_total_steps+1)

        self.G = minibatch.get_graph()
        self.model = model
        self.model_path = model_checkpoint
        self.sess = sess
        tf.reset_default_graph()
        sess.close()



    def infer_graph(self, data, random_context=True, model_path=None
                    , new_session=False, test_data=None
                    , update_model_graph=True):

        # reset to load old model and use new inference adjacency matrix.
        # in order to obtain trained weights
        if self.model is None:
            print(">")
            print(">>>> No known model...")
        print(">")
        model = self.model
        sess = self.sess
        #tf.reset_default_graph()
        graph = tf.get_default_graph()

        with graph.as_default():
            if model_path is not None:
                # path to saved model file
                self.model_path = model_path

            # graph information for inference graph

            self.G = data[0]
            self.features = data[1]
            self.id_map = data[2]
            self.class_map = data[4]
            G, features, id_map, class_map = (self.G, self.features, self.id_map, self.class_map)
            if isinstance(list(self.class_map.values())[0], list):
                num_classes = len(list(self.class_map.values())[0])
            else:
                num_classes = len(set(self.class_map.values()))

            if not self.features is None:
                # pad with dummy zero vector
                features = np.vstack([self.features, np.zeros((self.features.shape[1],))])

            FLAGS = self.FLAGS

            context_pairs = data[3] if random_context else None

            batch_infer = None
            if update_model_graph is True:
                self.test_nodes = [n for n in self.G.nodes()]# if self.G.node[n]['test']]
                batch_infer = [self.id_map[n] for n in self.test_nodes]
                FLAGS.batch_size = len(batch_infer)
                adj = len(self.id_map) * np.ones((len(self.id_map) , FLAGS.max_degree))
                inference_labels = np.vstack([
                    self._make_label_vec(node,self.class_map,num_classes) for node in batch_infer])
                FLAGS.batch_size = len(self.test_nodes)#adj.shape[0]
                FLAGS.validate_batch_size = adj.shape[0]
                print(" >>>>> Adjacency shape: ", adj.shape[0])
                print(" >> size test set ", len(self.test_nodes))
                placeholders = self.fetch_placeholders(batch=batch_infer, labels=inference_labels)
                    #= self.construct_placeholders(num_classes, batch=self.test_nodes,
                    #                                       labels = inference_labels,
                    #                                       inf_batch_shape=FLAGS.validate_batch_size,#FLAGS.batch_size,
                    #                                       class_map=class_map, id_map=id_map,
                    #                                       name_prefix='inf', infer=True)
                self.placeholders = placeholders
            else:
                placeholders = self.placeholders
            inference = True
            minibatch = NodeMinibatchIterator(self.G,
                                              self.id_map,
                                              self.placeholders,
                                              self.class_map,
                                              num_classes,
                                              batch_size=FLAGS.batch_size,
                                              max_degree=FLAGS.max_degree,
                                              context_pairs=context_pairs,
                                              train=not inference)

            adj_info_ph = self.adj_info_ph#tf.placeholder(tf.int32, shape=minibatch.test_adj.shape)#, name="adj_placeholder")

            #adj_info =  tf.get_default_graph().get_tensor_by_name("adj_info:0")

            #adj_info = tf.get_variable("adj_info", shape=minibatch.test_adj.shape)#, validate_shape=False)# tf.Variable(adj_info_ph, trainable=False, name="inf_adj_info")
            adj_info = tf.Variable(initial_value=minibatch.test_adj, trainable=False, dtype=tf.int32)
            #inference_adj_info = tf.assign(adj_info, minibatch.test_adj)
            #adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape, name="inf_adj_placeholder")
            #adj_info_train = tf.Variable(adj_info_ph, trainable=False, name="inf_adj_info")

            print( "inference graph adjaceny shape: ", adj_info.shape)


            #
            # For inference model is assumed to be known and pre-trained
            # Assuming loading model from checkpoint file and the model hasn't
            # been made a class variable after training in the same process,
            # otherwise model from training will be used.
            if model is None:
                print("    >>>>> Creating new model...")
                if FLAGS.model == 'graphsage_mean':
                    # Create model
                    sampler = UniformNeighborSampler(adj_info)
                    if FLAGS.samples_3 != 0:
                        layer_infos = [SAGEInfo("node_gsmean", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                       SAGEInfo("node_gsmean", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                       SAGEInfo("node_gsmean", sampler, FLAGS.samples_3, FLAGS.dim_2)]
                    elif FLAGS.samples_2 != 0:
                        layer_infos = [SAGEInfo("node_gsmean", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                       SAGEInfo("node_gsmean", sampler, FLAGS.samples_2, FLAGS.dim_2)]
                    else:
                        layer_infos = [SAGEInfo("node_gsmean", sampler, FLAGS.samples_1, FLAGS.dim_1)]

                    model = SupervisedGraphsage(num_classes, placeholders,
                                                features,
                                                adj_info,
                                                minibatch.deg,
                                                layer_infos,
                                                model_size=FLAGS.model_size,
                                                sigmoid_loss=FLAGS.sigmoid,
                                                identity_dim=FLAGS.identity_dim,
                                                logging=True)
                elif FLAGS.model == 'gcn':
                    # Create model
                    sampler = UniformNeighborSampler(adj_info)
                    layer_infos = [SAGEInfo("node_gcn", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                                   SAGEInfo("node_gcn", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]

                    model = SupervisedGraphsage(num_classes, placeholders,
                                                features,
                                                adj_info,
                                                minibatch.deg,
                                                layer_infos=layer_infos,
                                                aggregator_type="gcn",
                                                model_size=FLAGS.model_size,
                                                concat=False,
                                                sigmoid_loss=FLAGS.sigmoid,
                                                identity_dim=FLAGS.identity_dim,
                                                logging=True)

                elif FLAGS.model == 'graphsage_seq':
                    sampler = UniformNeighborSampler(adj_info)
                    layer_infos = [SAGEInfo("node_gsseq", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                   SAGEInfo("node_gsseq", sampler, FLAGS.samples_2, FLAGS.dim_2)]

                    model = SupervisedGraphsage(num_classes, placeholders,
                                                features,
                                                adj_info,
                                                minibatch.deg,
                                                layer_infos=layer_infos,
                                                aggregator_type="seq",
                                                model_size=FLAGS.model_size,
                                                sigmoid_loss=FLAGS.sigmoid,
                                                identity_dim=FLAGS.identity_dim,
                                                logging=True)

                elif FLAGS.model == 'graphsage_maxpool':
                    sampler = UniformNeighborSampler(adj_info)
                    layer_infos = [SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                   SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]

                    model = SupervisedGraphsage(num_classes, placeholders,
                                                features,
                                                adj_info,
                                                minibatch.deg,
                                                layer_infos=layer_infos,
                                                aggregator_type="maxpool",
                                                model_size=FLAGS.model_size,
                                                sigmoid_loss=FLAGS.sigmoid,
                                                identity_dim=FLAGS.identity_dim,
                                                logging=True)

                elif FLAGS.model == 'graphsage_meanpool':
                    sampler = UniformNeighborSampler(adj_info)
                    layer_infos = [SAGEInfo("node_meanpool", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                   SAGEInfo("node_meanpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]

                    model = SupervisedGraphsage(num_classes, self.placeholders,
                                                features,
                                                adj_info,
                                                minibatch.deg,
                                                layer_infos=layer_infos,
                                                aggregator_type="meanpool",
                                                model_size=FLAGS.model_size,
                                                sigmoid_loss=FLAGS.sigmoid,
                                                identity_dim=FLAGS.identity_dim,
                                                logging=True)

                else:
                    raise Exception('Error: model name unrecognized.')
                config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
                config.gpu_options.allow_growth = True
                # config.gpu_options.per_process_gpu_memory_fraction = self.GPU_MEM_FRACTION
                config.allow_soft_placement = True

                # Initialize session
                sess = tf.Session(config=config)
            elif update_model_graph:
                sess = self.sess
                print(">>>> Updating known model with new adjacency matrix")
                sampler = UniformNeighborSampler(adj_info, name="inf_neighbor_sampler")
                if FLAGS.model == 'graphsage_mean':
                    if FLAGS.samples_3 != 0:
                        layer_infos = [SAGEInfo("node_gsmean", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                       SAGEInfo("node_gsmean", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                       SAGEInfo("node_gsmean", sampler, FLAGS.samples_3, FLAGS.dim_2)]
                    elif FLAGS.samples_2 != 0:
                        layer_infos = [SAGEInfo("node_gsmean", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                       SAGEInfo("node_gsmean", sampler, FLAGS.samples_2, FLAGS.dim_2)]
                    else:
                        layer_infos = [SAGEInfo("node_gsmean", sampler, FLAGS.samples_1, FLAGS.dim_1)]
                elif FLAGS.model == 'gcn':
                    layer_infos = [SAGEInfo("node_gcn", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                                   SAGEInfo("node_gcn", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]
                elif FLAGS.model == 'graphsage_seq':
                    layer_infos = [SAGEInfo("node_gsseq", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                   SAGEInfo("node_gsseq", sampler, FLAGS.samples_2, FLAGS.dim_2)]

                elif FLAGS.model == 'graphsage_maxpool':
                    layer_infos = [SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                   SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]

                elif FLAGS.model == 'graphsage_meanpool':
                    layer_infos = [SAGEInfo("node_meanpool", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                   SAGEInfo("node_meanpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
                else:
                    raise Exception('Error: model name unrecognized.')
                model.update_graph_adjacency( self.features, self.placeholders, adj_info
                                              , layer_infos, minibatch.deg, num_classes, identity_dim = 0
                                              , name='infer')

            sess = self.sess

            #load pre-trained
            if self.model_path is not None and self.model is None:
                print("    >>>> loading tf model from: ", self.model_path)

                import os.path
                print("     >> specifically: ", os.path.dirname(self.model_path) + '/')
                saver = tf.train.import_meta_graph(self.model_path+'.meta')
                saver.restore(sess,save_path=self.model_path)
                graph = tf.get_default_graph()

            # reset session graph to initialize new variables
            #merged = tf.compat.v1.summary.merge_all()

            # write inference model run to file
            ####summary_writer = tf.summary.FileWriter(self.log_dir(), sess.graph)

            """
            Running incremental eval instead: 
            # Init variables
            print("    >>>>>>  pre-init")"""
            #with self.sess.graph.as_default():
            #sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.test_adj})
            #sess.run(adj_info.initializer, feed_dict={adj_info_ph: minibatch.test_adj})

            sess.run(tf.global_variables_initializer())#, feed_dict={adj_info_ph: minibatch.adj})
            #sess.run(tf.initialize_all_variables(), feed_dict={adj_info_ph: minibatch.test_adj})

            # for inference a single batch is entire graph
            #inf_feed_dict, batch_infer, inference_labels = minibatch.inference_feed_dict() # batch1, .inference_feed_dict()

            ##inf_b1 = tf.Variable(self.placeholders['batch1'], dtype=tf.int32
            ##                    , shape=(len(batch_infer)),validate_shape=False
            ##                    , trainable=False, name='inf_batch1_name')
            #inf_b = tf.Variable(self.placeholders['batch'], dtype=tf.int32
            #                    , shape=(len(batch_infer)), validate_shape=False
            #                    , trainable=False, name='inf_batch_name')

            #inf_bs = tf.Variable(self.placeholders['batch_size'], dtype=tf.int32
            #                     , trainable=False,validate_shape=False, name='inf_bs_name')

            #inf_blabels = tf.Variable(self.placeholders['labels'], dtype=tf.float32
            #                          , shape=(len(inference_labels), num_classes), trainable=False
            #                          , validate_shape=False,name='inf_blab_name')
            ##inf_batch2labels = tf.Variable(self.placeholders['batch2'], dtype=tf.float32
            ##                          , shape=(len(batch_infer), num_classes), trainable=False
            ##                          , validate_shape=False, name='inf_batch2lab_name')

            #inf_dropout = tf.Variable(self.placeholders['dropout'], dtype=tf.float32, shape=(), validate_shape=False,trainable=False, name='inf_dpo')

            print(" >>> inf batch size: ", len(batch_infer))
            print(" >>> num labels: ", len(inference_labels))

            # b1 = tf.assign(b.initializer, placeholders['batch'])
            #sess.run(b.initializer, feed_dict={placeholders['batch']: inf_feed_dict['batch']})

            print(" >>>>>> After run sess -")

            # Infer with Model

            total_steps = 0
            avg_time = 0.0
            epoch_val_costs = []

            #train_adj_info = tf.assign(adj_info, minibatch.adj)

            #inf_batch_assign = tf.assign(inf_b, batch_infer)#, name='inf_batch_name')
            ##inf_batch1_assign = tf.assign(inf_b1, batch_infer)
            #inf_batch_size_assign = tf.assign(inf_bs, len(batch_infer))#, name='inf_batch_size')
            #inf_label_assign = tf.assign(inf_blabels, inference_labels)#, name='inf_labels')
            ##inf_batch2label_assign = tf.assign(inf_batch2labels, inference_labels)  # , name='inf_labels')
            #inf_dropout_assign = tf.assign(inf_dropout, FLAGS.dropout)

            if self.placeholders is None or update_model_graph is True:

                print("    >>>> Initialializing placeholders...")
                total_nodes = adj_info.shape[0]


            #sess.run(bs.initializer, feed_dict={placeholders['batch_size']:FLAGS.batch_size})
            # for forward pass during inference labeling doesn't matter
            # since loss is not utilized for updating the model.
            # For this reason we can label all as 'test' as done
            # in training and use labeling obtained from trained model.
            # Node labels are assigned to 'prediction' attribute of nodes
            # using minibatch_iter class's 'update_batch_prediction'
            # with infered predictions ('preds')

            #for epoch in range(FLAGS.epochs):
            ##minibatch.shuffle()

            # reset session graph to initialize new variables
            merged = tf.compat.v1.summary.merge_all()

            nieve_pass = False
            if nieve_pass:
                iter = 0
                print('Performing Inference Naive Forward Pass...')
                epoch_val_costs.append(0)
                while not minibatch.end():
                    # Construct feed dictionary
                    feed_dict, labels = minibatch.next_minibatch_feed_dict()
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                    t = time.time()
                    # Training step
                    print("    >>> evaluating minibatches...")
                    outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
                    train_cost = outs[2]

                    preds = self.pred_values(outs[-1])
                    minibatch.update_batch_prediction(preds)

                    # currently no need to perform inference on validation set however
                    # will keep here to later implement sanity checks to obtain f1 scores
                    # from classifications over graphs not used for training but with labelings
                    # unused for classification.
                    if iter % FLAGS.validate_iter == 0:
                        # Validation
                        #sess.run(val_adj_info.op)
                        ####sess.run(batch_assign.op)
                        ####sess.run(batch_size_assign.op)
                        if FLAGS.validate_batch_size == -1:
                            val_cost, val_f1_mic, val_f1_mac, duration = self.incremental_evaluate(sess, model, minibatch,
                                                                                              FLAGS.batch_size)
                        else:
                            val_cost, val_f1_mic, val_f1_mac, duration = self.evaluate(sess, model, minibatch,
                                                                                  FLAGS.validate_batch_size)
                        #sess.run(train_adj_info.op)
                        epoch_val_costs[-1] += val_cost
                    """ removed summary writer for now
                    if total_steps % FLAGS.print_every == 0:
                        summary_writer.add_summary(outs[0], total_steps)
                    """
                    # Print results
                    avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

                    if total_steps % FLAGS.print_every == 0:
                        train_f1_mic, train_f1_mac = self.calc_f1(labels, outs[-1])
                        print("If no ground truth labeling provided these are meaningless",
                              "Iter:", '%04d' % iter,
                              "train_loss=", "{:.5f}".format(train_cost),
                              "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                              "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                              #"val_loss=", "{:.5f}".format(val_cost),
                              #"val_f1_mic=", "{:.5f}".format(val_f1_mic),
                              #"val_f1_mac=", "{:.5f}".format(val_f1_mac),
                              "time=", "{:.5f}".format(avg_time))

                    iter += 1
                    total_steps += 1

                    if total_steps > FLAGS.max_total_steps:
                        break

                #if total_steps > FLAGS.max_total_steps:
                #    break
                # end epoch iter

            print(" >>>> Inference Forward Pass Finished")
            print(" >>>> Performing Incremental Evaluate...")
            # Will keep this for use later for validating experiments
            run_val = True
            if run_val==True:

                #inference_adj_info = tf.assign(adj_info, minibatch.test_adj)

                #train_adj_info = tf.assign(adj_info_train, minibatch.adj)
                # supervised prediction on validation nodes
                #if not nieve_pass:
                #

                #sess.run(inf_batch_assign.op)
                ## sess.run(inf_batch1_assign.op)
                #sess.run(inf_batch_size_assign.op)
                #sess.run(inf_label_assign.op)
                ## sess.run(inf_batch2label_assign.op)
                #sess.run(inf_dropout_assign.op)

                feed_dict_inf = dict()
                feed_dict_inf.update({'batch_size:0' : self.placeholders['batch_size']})  # inf_bs})#inf_bs,#len(batch1)})
                ## feed_dict_inf.update({self.placeholders['batch1']: batch_infer})#inf_b})#inf_b})
                feed_dict_inf.update({'dropout:0' : self.placeholders['dropout']})#inf_dropout})
                feed_dict_inf.update({'labels:0' : self.placeholders['labels']})#inf_blabels})
                ## feed_dict_inf.update({self.placeholders['batch2']: inference_labels})
                feed_dict_inf.update({'batch:0' : self.placeholders['batch']})

                #feed_dict_inf.update({self.placeholders['batch_size']: len(batch_infer)})#inf_bs})#inf_bs,#len(batch1)})
                ## feed_dict_inf.update({self.placeholders['batch1']: batch_infer})#inf_b})#inf_b})
                #feed_dict_inf.update({self.placeholders['dropout']:FLAGS.dropout})#inf_dropout})
                #feed_dict_inf.update({self.placeholders['labels']: inference_labels})#inf_blabels})
                ## feed_dict_inf.update({self.placeholders['batch2']: inference_labels})
                #feed_dict_inf.update({self.placeholders['batch']: batch_infer})

                #sess.run(inf_batch_assign.op)
                #sess.run(inf_batch1_assign.op)
                #sess.run(inf_batch_size_assign.op)
                #sess.run(inf_label_assign.op)
                #sess.run(inf_batch2label_assign.op)
                #sess.run(inf_dropout_assign.op)

                #minibatch.placeholders = self.placeholders#inference_placeholders#feed_dict_inf


                #sess.run(train_adj_info.op)
                """
                feed_dict_inf = {inf_bs: len(batch_infer),  # inf_bs})#inf_bs,#len(batch1)})
                    inf_b: batch_infer,   # inf_b})#inf_b})
                    inf_dropout: FLAGS.dropout,  # inf_dropout})
                    inf_blabels: inf_labels}  # inf_blabels})
    
                feed_dict_inf = {inf_batch_size_assign : len(batch_infer),
                                 inf_batch_assign : batch_infer,
                                 inf_dropout_assign : FLAGS.dropout,
                                 inf_label_assign : inf_labels}
                """

                #sess.run(inference_adj_info.op)

                ####val_cost, val_f1_mic, val_f1_mac, duration = self.incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
                """
                print("Full validation stats:",
                      "loss=", "{:.5f}".format(val_cost),
                      "f1_micro=", "{:.5f}".format(val_f1_mic),
                      "f1_macro=", "{:.5f}".format(val_f1_mac),
                      "time=", "{:.5f}".format(duration))
                with open(self.log_dir() + "val_stats.txt", "w") as fp:
                    fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
                             format(val_cost, val_f1_mic, val_f1_mac, duration))
                """
                print("Writing inference set stats to file (don't peak!)")

                #outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict_inf)

                #node_outs_val = sess.run([model.preds, model.loss], feed_dict=feed_dict_inf)


                val_cost, val_f1_mic, val_f1_mac, duration = self.incremental_evaluate(
                    sess, model, minibatch, len(batch_infer), test=True, inference=True,
                    infer_feed_dict=feed_dict_inf, infer_labels=inference_labels)
                #test=True)# #(placeholders, inf_labels))
                with open(self.log_dir() + "test_stats.txt", "w") as fp:
                    fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                             format(val_cost, val_f1_mic, val_f1_mac))

                #
                # unsupervised prediction and embedding of validation nodes
                #

            self.inference_G = minibatch.get_graph()

            # return graph with nodes labeled by predictions
            # from pre-trained model.
            return self.inference_G


    def get_graph(self):
        return self.G

    def get_graph_prediction(self):
        return self.inference_G

    def format_msc_feature_graph(self, image, msc, mask, segmentation, persistence, blur):

        mscgnn = MSCGNN(msc=msc)
        # add number id to name
        msc_graph_name = 'msc-feature-graph-' + str(persistence) + 'blur-' + str(blur)
        mscgnn.msc_feature_graph(image=np.transpose(np.mean(image, axis=1), (1, 0)), X=image.shape[0], Y=image.shape[2]
                                 , validation_samples=2, validation_hops=20
                                 , test_samples=0, test_hops=0, accuracy_threshold=0.2
                                 , write_json_graph_path='./data', name=msc_graph_name
                                 , test_graph=False)

        if self.load_walks:
            print('... Generating Random Walk Neighborhoods for Node Co-Occurance')
            walk_embedding_file = os.path.join(self.LocalSetup.project_base_path, 'datasets', 'walk_embeddings'
                                               , str(persistence) + str(blur) + 'test_walk')
            random_walk_embedding(mscgnn.G, walk_length=4, number_walks=10, out_file=walk_embedding_file)

        G, feats, id_map \
            , walks, class_map \
            , number_negative_samples \
            , number_positive_samples = format_data(dual=self.G
                                                    , features=self.features
                                                    , node_id=self.node_id
                                                    , id_map=self.node_id
                                                    , node_classes=self.node_classes
                                                    , train_or_test=''
                                                    , scheme_required=True
                                                    , load_walks=self.load_walks)
        training_sample = (G, feats, id_map, walks
                           , class_map, number_negative_samples, number_positive_samples)
        return training_sample

    def batch_train(self, msc_collection, test_data=None):
        mscbatch = []

        training_msc_dataset = msc_collection[0]
        persistence_values = msc_collection[1]
        blur_sigmas = msc_collection[2]
        for image, msc_set, mask, segmentation in training_msc_dataset:
            pers=0
            blur=0

            # add number id to name
            if self.val_model == 'cvt':
                msc = msc_set[(sorted(persistence_values)[pers], blur_sigmas[blur])]
                mscgnn = MSCGNN(msc=msc, msc_collection=msc_collection)
                msc_graph_name = 'msc-feature-graph-' + str(sorted(persistence_values)[pers]) + 'blur-' + str(blur)
                mscgnn.msc_feature_graph(image=np.transpose(np.mean(image, axis=1), (1, 0)), X=image.shape[0],
                                         Y=image.shape[2]
                                         , validation_samples=2, validation_hops=20
                                         , test_samples=0, test_hops=0, accuracy_threshold=0.2
                                         , write_json_graph_path='./data', name=msc_graph_name
                                         , test_graph=False)
            else:
                msc = msc_set[(sorted(persistence_values)[0], blur_sigmas[blur])]
                mscgnn = MSCGNN(msc=msc, msc_collection=msc_collection)
                if supervised:
                    msc_graph_name = 'sup-msc-feature-graph-trainPers-' + str(
                        sorted(persistence_values)[-1]) + 'valPers-' + str(
                        sorted(persistence_values)[0]) + '-blur-' + str(
                        blur_sigmas[blur])
                else:
                    msc_graph_name = 'unsup-msc-feature-graph-' + str(sorted(persistence_values)[0]) + 'blur-' + str(
                        blur_sigmas[blur])
                mscgnn.msc_feature_graph(image=np.transpose(np.mean(image, axis=1), (1, 0)), X=image.shape[0],
                                         Y=image.shape[2]
                                         , persistence_values=persistence_values, blur=blur_sigmas[blur]
                                         , val_model='persistence_subset'
                                         , test_samples=0, test_hops=0, accuracy_threshold=0.2
                                         , write_json_graph_path='./data', name=msc_graph_name
                                         , test_graph=False)
            if self.load_walks:
                print('... Generating Random Walk Neighborhoods for Node Co-Occurance')
                walk_embedding_file = os.path.join(self.LocalSetup.project_base_path, 'datasets', 'walk_embeddings'
                                                   , str(self.persistence_values[pers]) + str( blur_sigmas[blur]) + 'test_walk')
                random_walk_embedding(mscgnn.G, walk_length=4, number_walks=10, out_file=walk_embedding_file)

            G, feats, id_map \
                , walks, class_map \
                , number_negative_samples \
                , number_positive_samples = format_data(dual=mscgnn.G
                                                        , features=mscgnn.features
                                                        , node_id=mscgnn.node_id
                                                        , id_map=mscgnn.node_id
                                                        , node_classes=mscgnn.node_classes
                                                        , train_or_test=''
                                                        , scheme_required=True
                                                        , load_walks=walk_embedding_file)
            training_sample = (G, feats, id_map, walks
                               , class_map, number_negative_samples, number_positive_samples)

            print(" %%%%% feature graph complete")




            self._train(training_sample[:-2])

    def get_data(self, train_data=None):
        print("Loading training data..")
        #train_data = load_data(FLAGS.train_prefix)

        ### load MSC data
        print('loading msc graph data')

        self.generate_embedding = self.use_embedding is not None
        if self.use_embedding is not None:
            self.G = self.use_embedding[0]  # graph to embed
            self.feats = self.use_embedding[1]  # features of graph to embed
            self.id_map = self.use_embedding[2]  # learned embedding id map
            self.walks = self.use_embedding[3] if self.load_walks is None else []
            self.class_map = []
        # Collect training data
        # train from saved file, assumes pre-labeled train/test nodes
        if self.msc_collection is None:
            if self.train_prefix and self.G is None:
                print("loading graph data for gnn training")
                self.train_prefix = self.train_prefix

                train_data = load_data(self.train_prefix, load_walks=self.load_walks, scheme_required=True,
                                       train_or_test='train')

                self.number_negative_samples = train_data[len(train_data) - 2]
                self.number_positive_samples = train_data[len(train_data) - 1]
                number_samples = self.number_negative_samples + self.number_positive_samples
                proportion_negative = int(number_samples / float(self.number_negative_samples))

            # train from passed graph, assumed pre-labeled(/processed)
            # graph with test/train nodes
            elif self.G is not None and self.feats is not None and self.id_map is not None and self.class_map is not None and not self.train_prefix:
                print(">>>")
                print("first elif")
                print(">>>>")
                train_prefix = 'nNeg-' + str(self.number_negative_samples) + 'nPos-' + str(self.number_positive_samples)
                print("using pre-processed graph data for gnn training")
                self.number_negative_samples = self.number_negative_samples
                number_samples = self.number_negative_samples + self.number_positive_samples
                proportion_negative = int(number_samples / float(self.number_negative_samples))
                train_data = (self.G, self.feats, self.id_map, self.walks, self.class_map, [], [])

            # train from cvt sampled graph and respective in/out arcs as train
            #elif self.positive_arcs and self.negative_arcs:
            #    train_data = load_data(self.positive_arcs, self.negative_arcs, load_walks=self.load_walks, scheme_required=True,
            #                           train_or_test='train')
            #    self.number_negative_samples = len(self.negative_arcs)
            #    number_samples = len(self.positive_arcs) + len(self.negative_arcs)
            #    proportion_negative = int(number_samples / float(self.number_negative_samples))

            # keep labeled (test/train) graph for later use in testing
            self.graph = train_data[0]
            self.features = train_data[1]
            self.id_map = train_data[2]
            self.node_classes = train_data[4]

            if self.load_walks:
                walks = []
                if isinstance(self.graph.nodes()[0], int):
                    conversion = lambda n: int(n)
                else:
                    conversion = lambda n: n
                with open(self.load_walks + "-walks.txt") as fp:
                    for line in fp:
                        walks.append(map(conversion, line.split()))

            train_data = (
            train_data[0], train_data[1], train_data[2], walks, train_data[4], train_data[5], train_data[6])
        return train_data


"""    def main(self, argv=None):
        print("Loading training data..")
        #train_data = load_data(FLAGS.train_prefix)

        ### load MSC data
        print('loading msc graph data')

        train_data=self.get_data()
        ## use msc graph data
        # begin training
        print('Begin GNN training')
        print('')
        if self.msc_collection is None:
            self._train(train_data[:-2])
        # else:
        #    self.batch_train(self.msc_collection)
        print("Done loading training data..")
        #_train(train_data)

        if __name__ == '__main__':
            tf.app.run()"""

