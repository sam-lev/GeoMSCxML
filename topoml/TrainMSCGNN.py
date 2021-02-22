
import sys
import os
import numpy as np
import copy
from skimage import io
import imageio
import argparse
from topoml.MSCGNN import MSCGNN
from topoml.topology.mscnn_segmentation import mscnn_segmentation
from topoml.topology.MSCSegmentation import MSCSegmentation
from topoml.topology.geometric_msc import GeoMSC
from topoml.topology.geometric_msc import GeoMSC_Union
from topoml.ml.MSCLearningDataSet import MSCRetinaDataset
from topoml.ui.LocalSetup import LocalSetup
from topoml.ml.MSCSample import MSCSample
from topoml.graphsage.utils import random_walk_embedding

# sys.path.append(os.getcwd())



parser = argparse.ArgumentParser()
parser.add_argument('--gpu',default='0', type=str, help="coma separated list of gpus to use")
parser.add_argument('--env', default='multivax',  type=str, help="system to run, if multivax hard assign gpu, sci assumes slurm")
args = parser.parse_args()

def note(log=""):
    print(log)

class MSCGNNTrainer:
    def __init__(self, collect_datasets=False, compute_msc=False, test_param=False):

        note("Comparison with normalization from jan9 im pred, dif being normalization /  better hyperparam, polarity 24 , -3 pow pers, (4 im, 4 feat, 20 walk 5 len, internal eye predicted farther from optic vessel graph ")



        # declarations for running examples
        #for persistence subgraph test, val train, three pers needed and consecutively nested
        #self.persistence_values = sorted([0.001,0.001,0.001])   # , 0.01, 0.1]#[10, 12, 15, 20 , 23, 25, 30] # below 1 for GeoMSC


        self.number_images = 1#4
        self.min_number_features = 1
        self.number_features = 3

        #list of persistences to use for image set in geomsc union space
        #dict is {index : number images, index+1 : number imaages, ... }
        self.persistence_values = sorted([1.01e-7])#, 1.01e-2])#, 1.01e-2, 1.01e-1])
        self.persistence_cardinality = {0: 1}#3, 1:1}#, 1: 2, 2:1}#self.number_images}
        self.blur_sigmas = [1]  # 1.0, 2.0, 5.0, 10] # add iterative blur &  try multichannel

        # index for respective images to be first train image and inference image
        self.train_data_idx = '0'
        self.inference_data_idx = '0'#'1'#str(self.number_images-1) #'1'

        
        # index od pers for first trainng image and inference image
        self.pers_train_idx = 0
        self.pers_inf_idx = 0
        self.blur = 0

        self.select_label = True
        #                                 1,
        supervised_learning_rates = [1.0e-1, 1.0e-2, 1.01e-3, 3.0e-4, 1.0e-5]
        unsupervised_learning_rates = [2.0e-1, 2.0e-6, 2.0e-7, 2.0e-8]

        self.learning_rate = supervised_learning_rates[2]#1.01e-4
        self.weight_decay = 0#.0
        #                [   0,   1,    2, 3, 4, 5,  6,  7,  8,  9, 10, 11]
        polarity_study = [-0.8, 0.25, 0.8, 1, 2, 4 ,7, 10, 18, 23, 25, 32] #10 > 23 #neg samples along walk context pair, 2 bc two min max
        self.polarity = polarity_study[9] #23 opt so far

        self.epochs = 20#60
        self.depth = 2

        # to ensure in line graph random walks extend to neighbors, 3 and more walks to improve statistics
        # walks used to define node stochastic node similarity based on observed co-occurance along
        # numerous walks.
        self.walk_length = 3
        self.number_walks = 55
        self.validation_samples = 4
        self.validation_hops = 4
        
        self.batch_size = 64

        # each layer is a hop, so l2 is 2-hop, this is the number of samples taken from two hops
        # l1 is then k-hop, sample higher depth 2-hop neighbors first then 1 hop
        # sample farther and generate embedding towards target node / 1-hop
        # l1 = k-hops, lk = 1 hop
        self.max_node_degree =  37#subsample edges so no node has degree larger than <---
        self.degree_l1       =  6  # number of neighbors sampled per node (can not be greater than msx_node_degree)
        self.degree_l2       =  3# edge chromatic number of G is equal to vertex chromatic number L(G)
        self.degree_l3       =  0

        self.aggregator = \
            ['graphsage_maxpool', 'graphsage_seq', 'graphsage_mean', 'gcn'
                , 'graphsage_meanpool', 'n2v'][0]

        self.model_size      =  "small"
        self.out_dim_1       =  256 #512//2 #int(256 / 2)  #
        self.out_dim_2       =  256 #512//2 #int(256 / 2)

        self.all_param = [ self.persistence_values, self.blur_sigmas , self.number_images ,self.number_features, self.train_data_idx , self.inference_data_idx , self.learning_rate ,  self.weight_decay,
        self.polarity ,self.epochs ,  self.depth ,self.walk_length , self.number_walks , self.validation_samples , self.validation_hops , self.batch_size ,self.max_node_degree,self.degree_l1 ,self.degree_l2 , self.degree_l3 ,self.model_size,  self.out_dim_1 , self.out_dim_2  ]

        if test_param == True:
            self.number_images = 1
            self.min_number_features = 1
            self.number_features = 3

            self.persistence_values = sorted([1.01e-3])#, 1.01e-2])  # , 1.01e-2, 1.01e-1])
            self.persistence_cardinality = {0: 1}#, 1: 2}  # , 1: 2, 2:1}#self.number_images}
            self.blur_sigmas = [1]

            # index for respective images to be first train image and inference image
            self.train_data_idx = '0'
            self.inference_data_idx = '0'

            # index od pers for first trainng image and inference image
            self.pers_train_idx = 0
            self.pers_inf_idx = 0
            self.blur = 0

            supervised_learning_rates = [1.0e-1, 1.0e-2, 1.1e-3, 3.0e-4]
            unsupervised_learning_rates = [2.0e-1, 2.0e-6, 2.0e-7, 2.0e-8]

            self.learning_rate = supervised_learning_rates[3]
            self.weight_decay = 0
            #                [   0,   1,    2, 3, 4, 5,  6,  7,  8,  9, 10]
            polarity_study = [-0.8, 0.25, 0.8, 1, 2, 3, 10, 17, 23, 25,
                              32]
            self.polarity = polarity_study[8]

            self.epochs = 12
            self.depth = 2

            self.walk_length = 5
            self.number_walks = 15
            self.validation_samples = 4
            self.validation_hops = 2

            self.batch_size = 64
            self.max_node_degree = 34  # subsample edges so no node has degree larger than <---
            self.degree_l1 = 32  # number of neighbors sampled per node (can not be greater than msx_node_degree)
            self.degree_l2 = 6  # edge chromatic number of G is equal to vertex chromatic number L(G)
            self.degree_l3 = 0
            self.aggregator = \
                ['graphsage_maxpool', 'graphsage_seq', 'graphsage_mean', 'gcn'
                    , 'graphsage_meanpool', 'n2v'][0]

            self.model_size = "small"
            self.out_dim_1 = 256//2# // 4  # 512//2 #int(256 / 2)  #
            self.out_dim_2 = 256//2# // 4  # 512//2 #int(256 / 2)
            self.all_param = [self.persistence_values, self.blur_sigmas, self.number_images, self.number_features,
                              self.train_data_idx, self.inference_data_idx, self.learning_rate, self.weight_decay,
                              self.polarity, self.epochs, self.depth, self.walk_length, self.number_walks,
                              self.validation_samples, self.validation_hops, self.batch_size, self.max_node_degree,
                              self.degree_l1, self.degree_l2, self.degree_l3, self.model_size, self.out_dim_1,
                              self.out_dim_2]

        self.msc_arc_accuracy_threshold = 0.1

        self.LocalSetup = LocalSetup(env=args.env)

        # Load the Retna data set images and hand segmentations
        # (training,test both Stare and Drive), map masks, and reformat images
        # to be same size
        self.collect_datasets = collect_datasets
        # run the Geometric MSC segmentation over images
        self.compute_msc = compute_msc
        self.drive_training_msc, self.drive_test_msc, self.stare_msc = None, None, None


    def _collect_datasets(self):
        print(" %%% collecting data buffers")
        self.MSCRetinaDataSet = MSCRetinaDataset(with_hand_seg=True)
        # get each set independently for msc computation
        self.drive_training_retina_array = self.MSCRetinaDataSet.get_retina_array(partial=False, msc=False
                                                                        , drive_training_only=True
                                                                        , env=args.env)
                                                                        #, number_images=number_images)
        ##drive_test_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False, drive_test_only=True)
        ##stare_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False, stare_only=True)

        # dataset buffers to use for training:
        self.drive_training_dataset = MSCRetinaDataset(self.drive_training_retina_array, split=None
                                                       , do_transform=False, with_hand_seg=True, shuffle=False)
        # dataset to use for validation
        ##drive_test_dataset = MSCRetinaDataset(drive_test_retina_array, split = None, do_transform = False, with_hand_seg=True)
        # dataset to use for testing
        ##stare_dataset = MSCRetinaDataset(stare_retina_array, split = None,do_transform = False, with_hand_seg=True)
        print("%collection complete")


    def _compute_msc(self):
        self.MSCSegmentation = MSCSegmentation()
        print(" %%% computing geometric msc")
        self.drive_training_dataset = self.MSCSegmentation.geomsc_segment_images(persistence_values=self.persistence_values
                                                                       , blur_sigmas=self.blur_sigmas
                                                                       , data_buffer=self.drive_training_dataset
                                                                       , data_path=self.LocalSetup.drive_training_path
                                                                       ,segmentation_path=self.LocalSetup.drive_training_segmentation_path
                                                                       , write_path=self.LocalSetup.drive_training_base_path
                                                                       , label=True#not self.select_label
                                                                       , save=False
                                                                       , valley=True, ridge=True
                                                                       ,env=args.env
                                                                       ,number_images=self.number_images
                                                                       , persistence_cardinality = self.persistence_cardinality)

        """drive_test_dataset = MSCSegmentation.geomsc_segment_images(persistence_values = persistence_values, blur_sigmas = blur_sigmas
                                              , data_buffer=drive_test_dataset, data_path = LocalSetup.drive_test_path, segmentation_path=LocalSetup.drive_test_segmentation_path
                                                          ,write_path = LocalSetup.drive_test_base_path, label=True, save=True, valley=True, ridge=True)
        stare_dataset = MSCSegmentation.geomsc_segment_images(persistence_values = persistence_values, blur_sigmas = blur_sigmas
                                              ,data_buffer=stare_dataset, data_path = LocalSetup.stare_training_data_path, segmentation_path=LocalSetup.stare_segmentations
                                                          ,write_path = LocalSetup.stare_base_path, label=True, save=True, valley=True, ridge=True)"""
        print("%geomsc complete")



    def collect_training_data(self):
        ## Get total retina array with newly computed msc
        ## and partition into train, test and val
        if self.collect_datasets:
            self._collect_datasets()
        if self.compute_msc:
            self._compute_msc()
        print(" %%% performing data buffer train, validation, test split ")
        self.drive_training_images, self.drive_training_msc_collections, self.drive_training_masks, self.drive_training_segmentations \
            = list( zip(*self.drive_training_dataset))
        """drive_test_images, drive_test_msc_collections, drive_test_masks, drive_test_segmentations = list(zip(*drive_test_dataset))
        stare_images, stare_msc_collections, stare_masks, stare_segmentations = list(zip(*stare_dataset))
        retina_dataset = list(zip(drive_training_images + drive_test_images + stare_images
                                  ,drive_training_msc_collections + drive_test_msc_collections + stare_msc_collections
                                  ,drive_training_masks + drive_test_masks + stare_masks
                                  ,drive_training_segmentations + drive_test_segmentations + stare_segmentations))"""


        ##
        ##
        ##   %%%%%%%%%%%%%%%%%%%     Initialize MSCGNN begin training.           %%%%%%%%%%%%%%%%%%%
        ##                  split data into train , validation and test sets. Each MSC is also subgraph
        ##                  into validation test and training. Embeddings are iteratively learned across msc.
        ##
        """ temp """
        self.retina_dataset = self.drive_training_dataset
        self.MSCRetinaDataSet.retina_array = self.retina_dataset
        self.MSCRetinaDataSet.get_retina_array(partial=False, msc=self.compute_msc
                                               , number_images=self.number_images)
        self.train_dataloader = MSCRetinaDataset(self.retina_dataset, split=None
                                                 , shuffle=False, do_transform = False, with_hand_seg=True)
        #val_dataloader = MSCRetinaDataset(retina_dataset, split = "val", do_transform = False, with_hand_seg=True)
        #test_dataloader = MSCRetinaDataset(retina_dataset, split = "test", do_transform = False, with_hand_seg=True)
        print(" %%% data buffer split complete")

    def prettyPrint(self, items):
        print(" ")
        print(">>>> ")
        for i in items:
            print(" >  ", i )
        print(">>>> ")
        print(" ")

    def learn_embedding(self, supervised=True, unsupervised=False, val_model='cvt', model_name=''
                        , load_preprocessed=False,load_preprocessed_walks=False, write_msc=False
                        , collect_features=False, draw=False
                        , active_learning = True):

        supervised = not unsupervised
        union_geomsc = self.number_images > 1

        print(" %%%%% creating geomsc feature graph ")

        # for image, msc, mask, segmentation in train_dataloader:
        # declarations for running examples
        # , 0.01, 0.1]#[10, 12, 15, 20 , 23, 25, 30] # below 1 for GeoMSC
          # 1.0, 2.0, 5.0, 10] # add iterative blur &  try multichannel

        blur = self.blur
        # hyper-param for gnn
        learning_rate = self.learning_rate
        polarity = self.polarity
        weight_decay = self.weight_decay
        epochs = self.epochs
        depth = self.depth
        walk_length = self.walk_length
        number_walks = self.number_walks
        validation_samples = self.validation_samples
        validation_hops=self.validation_hops

        #if not load_preprocessed:
        self.collect_training_data()
        image, msc_collection, mask, segmentation = self.train_dataloader[int(self.train_data_idx)]
        image_copy_1 = copy.deepcopy(image)
        image_copy = copy.deepcopy(image)



        infer = True
        if infer:
            #inference_mscgnn = mscgnn  # MSCGNN()
            inference_image, inference_msc_collection, mask, segmentation = self.train_dataloader[
                int(self.inference_data_idx)]
            inference_image_copy = copy.deepcopy(inference_image)
            if not load_preprocessed:
                inference_msc = inference_msc_collection[(sorted(self.persistence_values)[self.pers_inf_idx]
                                                          , self.blur_sigmas[blur])]
                #inference_mscgnn.assign_msc(msc = inference_msc) #(msc=inference_msc, msc_collection=inference_msc_collection)
                #inference_mscgnn.msc_collection = inference_msc_collection
            inference_msc_graph_name = 'inference_msc-feature-graph-' + str(self.persistence_values[self.pers_inf_idx]) + 'blur-' + str(blur)



            inference_msc_graph_name = 'inference_msc-feature-graph' + self.inference_data_idx +'-' + str(self.pers_inf_idx) + 'blur-' + str(blur)

        write_path = self.LocalSetup.drive_training_base_path
        msc_seg_path = os.path.join(write_path, 'msc_seg')
        msc_seg_base = msc_seg_path


        msc_seg_path = os.path.join(msc_seg_path, model_name +
                                'im_' + str(self.number_images) +'blur_' + str(self.blur_sigmas[blur]) + 'persistence_' + str(
                                    sorted(self.persistence_values)[self.pers_inf_idx]))
        if not os.path.exists(msc_seg_path):
            os.makedirs(os.path.join(msc_seg_path))

        msc_info = os.path.join(msc_seg_path,
                                'polarity_' + str(self.polarity) +
                                'lr_' + str(self.learning_rate) +
                                'blur_' + str(self.blur_sigmas[blur]) + 'persistence_' + str(
                                    sorted(self.persistence_values)[self.pers_inf_idx]))
        with open(msc_info + '.txt', 'w+') as f:
            f.write(
                " [ self.persistence_values, self.blur_sigmas , self.number_images ,self.number_features, self.train_data_idx , self.inference_data_idx , self.learning_rate ,  self.weight_decay, self.polarity \n")
            f.write(
                "self.epochs ,  self.depth ,self.walk_length , self.number_walks , self.validation_samples , self.validation_hops , self.batch_size ,self.max_node_degree,self.degree_l1 ,self.degree_l2 \n")
            f.write(", self.degree_l3 ,self.model_size,  self.out_dim_1 , self.out_dim_2 ,]" + "\n")
            for i in self.all_param:
                f.write(str(i) + "\n")
            f.close()


        # simple name to
        #msc_seg_path = os.path.join(msc_seg_path,
        #                            'blur_' + str(self.blur_sigmas[blur]) + 'persistence_' + str(sorted(self.persistence_values)[pers]))

        msc_text = os.path.join(msc_seg_path,
                                self.inference_data_idx + 'Blur' + str(
                                    self.blur_sigmas[blur]) + 'pers' + str(
                                    sorted(self.persistence_values)[self.pers_inf_idx]) + '-MSC.tif')

        mscgnn = MSCGNN()

        # add number id to name
        if val_model=='cvt':



            msc_graph_name = 'msc-feature-graph' + model_name\
                             + 'im-'+str(self.number_images)+ '_prs-' + str(self.persistence_values[self.pers_inf_idx]) \
                             + '_blur-' + str(self.blur_sigmas[blur])


            if load_preprocessed:
                geomsc = GeoMSC_Union() if union_geomsc else GeoMSC()
                geomsc.read_from_file(msc_seg_path)
                msc = geomsc
                geomsc.geomsc = geomsc
                mscgnn = MSCGNN(geomsc=geomsc)
                im_copy = copy.deepcopy(image)
                mscgnn.msc_feature_graph(load_preprocessed=True , multiclass=False, manually_select_training=True
                                         ,image=np.transpose(np.mean(im_copy,axis=1))
                                         , X=image.shape[0], Y=image.shape[2]
                                         , collect_features=collect_features, write_json_graph_path='./data'
                                         , name=msc_graph_name)
                computed_features = mscgnn.features


            else:

                msc = msc_collection[(self.persistence_values[self.pers_train_idx], self.blur_sigmas[blur])]
                geomsc = GeoMSC(geomsc=msc)
                if infer and union_geomsc:
                    if union_geomsc:
                        geomsc = GeoMSC_Union(msc, inference_msc)

                    #else:
                    #    msc = GeoMSC(geomsc=msc)
                    #for i in range(self.persistence_cardinality[self.pers_train]):
                    #    msc_i = msc_collection[(self.persistence_values[self.pers_train], self.blur_sigmas[blur])]
                    #    msc_union.U(msc_i)
                    #    msc=msc_union

                    print("pers cardinality: ", self.persistence_cardinality)

                    if self.number_images > 2:
                        image_msc_set = []



                        for pidx, pers in enumerate(sorted(self.persistence_values)):#int(self.train_data_idx) + 1, int(self.inference_data_idx)):
                            for i in range(self.number_images):
                                print("image ", i)
                                pers_cap = self.persistence_cardinality[pidx]
                                if pers_cap <= 0 or i == self.inference_data_idx:
                                    continue
                                print(" >>>> ")
                                print("adding msc to union ", str(pers))
                                im_i, msc_collection_i, mask_i, segmentation_i = \
                                    self.train_dataloader[i]
                                set_i = [im_i, msc_collection_i, mask_i, segmentation_i]
                                image_msc_set.append(set_i)
                                msc_i = msc_collection_i[(pers, self.blur_sigmas[blur])]

                                msc_union.U( msc_i)
                                ######msc_union.connect_union()
                                msc = msc_union  # .geomsc
                                msc_union = msc_union
                                self.persistence_cardinality[pidx] = pers_cap - 1
                                # do inference union here

                mscgnn = MSCGNN(msc=geomsc, msc_collection=msc_collection)
                im_copy = copy.deepcopy(image)
                mscgnn.msc_feature_graph(image=np.transpose(np.mean(im_copy, axis=1), (1, 0)),
                                         multiclass=False,
                                         manually_select_training=True,
                                         X=image.shape[0], Y=image.shape[2]
                                         , validation_samples=validation_samples,
                                         validation_hops=validation_hops
                                         , test_samples=0, test_hops=0,
                                         accuracy_threshold=self.msc_arc_accuracy_threshold
                                         , write_json_graph_path='./data', name=msc_graph_name
                                         , test_graph=False, sigmoid=False
                                         , min_number_features=self.min_number_features
                                         ,number_features=self.number_features)

                # for preserving feature information for second or new inference
                computed_features = mscgnn.features

                if write_msc:
                    msc.write_msc(msc_seg_path)





            #self.total_features = mscgnn.number_features

            #self.prettyPrint(["total number of features", self.total_features])

        else:

            if supervised:
                msc_graph_name = 'sup-' + model_name +'-msc-feature-graph' + self.train_data_idx +'-trainPers-' + str(
                    sorted(self.persistence_values)[self.pers_inf_idx]) + 'valPers-' + str(sorted(self.persistence_values)[self.pers_inf_idx]) + '-blur-' + str(
                    self.blur_sigmas[blur])


            else:
                msc_graph_name = 'unsup-' + model_name +'-msc-feature-graph' + self.train_data_idx \
                                 +'-' + str(sorted(self.persistence_values)[self.pers_inf_idx]) + 'blur-' + str(self.blur_sigmas[blur])

            msc_path_and_name = os.path.join(msc_seg_path,
                                             'partitions-Blur' + str(self.blur_sigmas[blur]) + 'pers' + str(
                                                 sorted(self.persistence_values)[self.pers_inf_idx]) + '-MSC.tif')
            if load_preprocessed:

                msc = GeoMSC_Union()
                msc.read_from_file(msc_seg_path)
                mscgnn = MSCGNN(msc=msc)
                mscgnn.msc_feature_graph(load_preprocessed=True
                                         , image=np.transpose(np.mean(image, axis=1)), X=image.shape[0],
                                         Y=image.shape[2]
                                         , write_json_graph_path='./data', name=msc_graph_name)
            else:
                # do inference graph union here

                msc = msc_collection[(sorted(self.persistence_values)[self.pers_train_idx], self.blur_sigmas[blur])]

                if infer:
                    msc_union = GeoMSC_Union(msc, inference_msc)
                    #####msc_union.connect_union()
                    msc = msc_union  # .geomsc

                    # for i in range(self.persistence_cardinality[self.pers_train]):
                    #    msc_i = msc_collection[(self.persistence_values[self.pers_train], self.blur_sigmas[blur])]
                    #    msc_union.U(msc_i)
                    #    msc=msc_union

                    if self.number_images > 2:
                        image_msc_set = []

                        for pers_idx in self.persistence_cardinality.keys():
                            max_pers_set = self.persistence_cardinality[pers_idx]
                            # for nump in range(max_pers_set):
                            pers_v = self.persistence_values[pers_idx]
                            for i in range(self.number_images):  # int(self.train_data_idx) + 1, int(self.inference_data_idx)):
                                if max_pers_set > 0 and i != self.inference_data_idx:
                                    im_i, msc_collection_i, mask_i, segmentation_i = \
                                        self.train_dataloader[i]
                                    set_i = [im_i, msc_collection_i, mask_i, segmentation_i]
                                    image_msc_set.append(set_i)
                                    msc_i = msc_collection_i[(pers_v, self.blur_sigmas[blur])]

                                    msc_union.U(msc_i)
                                    ######msc_union.connect_union()
                                    msc = msc_union  # .geomsc
                                    msc_union = msc_union
                                max_pers_set = max_pers_set - 1
                                # do inference union here


                mscgnn = MSCGNN(msc=msc, msc_collection=msc_collection)
                mscgnn.msc_collection = msc_collection
                mscgnn.msc_feature_graph(image=np.transpose(np.mean(image,axis=1),(1,0)), X=image.shape[0], Y=image.shape[2]
                                            ,persistence_values=self.persistence_values,blur=self.blur_sigmas[blur]
                                         ,val_model='persistence_subset'
                                            , test_samples=0, test_hops=0, accuracy_threshold=self.msc_arc_accuracy_threshold
                                            ,write_json_graph_path='./data', name=msc_graph_name
                                            ,test_graph=False)

                #self.total_features = mscgnn.number_features

                if draw and not infer:
                    mscgnn.draw_segmentation(image=image_copy
                                             ,path_name=msc_path_and_name
                                             ,type='partitions')


        print(" %%%%% feature graph complete")

        # construct unsupervised gnn model
        aggregator = self.aggregator

        print('... Beginning training with aggregator: ', aggregator)

        # Random walks used to determine node pairs for unsupervised loss.
        # to make a random walk collection use ./topoml/graphsage/utils.py
        # example run: python3 topoml/graphsage/utils.py ./data/json_graphs/test_ridge_arcs-G.json ./data/random_walks/full_msc_n-1_k-40
        ##load_walks='full_msc_n-1_k-40'
        ##load_walks='full_msc_n-1_k-50_Dp-10_nW-100'

        print('... Generating Random Walk Neighborhoods for Node Co-Occurance')
        walk_embedding_file = os.path.join(self.LocalSetup.project_base_path,'datasets','walk_embeddings'
                                           ,model_name\
                             + 'im_'+str(self.number_images)+ '_nWalk-'+str(self.number_walks)+'_walkLength-'+str(self.walk_length)
                             +self.train_data_idx+str(self.persistence_values[self.pers_inf_idx])
                           +str(self.blur_sigmas[0]) +'_walk')

        if not load_preprocessed_walks:
            random_walk_embedding(mscgnn.G, walk_length=walk_length, number_walks=number_walks, out_file=walk_embedding_file)

        if unsupervised:
            mscgnn.unsupervised(aggregator=aggregator, env=args.env)
        if supervised and not unsupervised:
            mscgnn.supervised(aggregator=aggregator, env=args.env)



        # file name for embedding of trained model
        embedding_name = 'msc-embedding_'+msc_graph_name #-pers-'+pers_string+'blur-'+str(self.blur_sigmas[blur])
        #'n-' + str(cvt[0]) + '_k-' + str(cvt[1]) + '_lr-' + str(learning_rate) + 'Plrty-' + str(
           #polarity) + '_epochs-' + str(epochs) + '_depth-' + str(depth) + 'trainGrp-' + train_group_name

        mscgnn.train( embedding_name = embedding_name, load_walks=walk_embedding_file
                      , learning_rate=learning_rate, epochs=epochs, batch_size=self.batch_size
                      , weight_decay=weight_decay, polarity=polarity
                      , depth=depth, gpu=args.gpu, val_model=val_model, sigmoid=False
                      , max_degree=self.max_node_degree, degree_l1=self.degree_l1, degree_l2=self.degree_l2 , degree_l3=self.degree_l3

                      , model_size = self.model_size
                            ,out_dim_1 = self.out_dim_1
                            , out_dim_2 = self.out_dim_2)

        if unsupervised and infer:

            mscgnn = mscgnn.classify(MSCGNN_infer=mscgnn, MSCGNN=mscgnn,
                                                   embedding_path_name=None# embedding_name
                                                   , embedding_prefix=embedding_name, learning_rate=learning_rate,
                                                   aggregator=aggregator, supervised=supervised)

        #if supervised:
        G = mscgnn.get_graph()
        #if loading then no MSC to equate, does nothing
        mscgnn.equate_graph(G)

        if active_learning:
            im_copy = copy.deepcopy(image)

            # for when adding new msc for inference
            #mscgnn = MSCGNN(msc=msc, msc_collection=msc_collection)

            #mscgnn.compiled_features = computed_features

            mscgnn.update_training_from_inference(image=np.transpose(np.mean(im_copy, axis=1), (1, 0)),
                                         multiclass=False,
                                         manually_select_training=False,
                                         X=image.shape[0], Y=image.shape[2]
                                         , validation_samples=validation_samples,
                                         validation_hops=validation_hops
                                         , test_samples=0, test_hops=0,
                                         accuracy_threshold=self.msc_arc_accuracy_threshold
                                         , write_json_graph_path='./data', name=msc_graph_name
                                         , test_graph=False, sigmoid=False
                                         , min_number_features=self.min_number_features
                                         ,number_features=self.number_features)

            print(' >>>> Re-running random walks for new context pairs')

            random_walk_embedding(mscgnn.G, walk_length=walk_length, number_walks=number_walks,
                                  out_file=walk_embedding_file)

            if unsupervised:
                mscgnn.unsupervised(aggregator=aggregator, env=args.env)
            if supervised and not unsupervised:
                mscgnn.supervised(aggregator=aggregator, env=args.env)

            mscgnn.train(embedding_name=embedding_name, load_walks=walk_embedding_file
                         , learning_rate=learning_rate, epochs=epochs, batch_size=self.batch_size
                         , weight_decay=weight_decay, polarity=polarity
                         , depth=depth, gpu=args.gpu, val_model=val_model, sigmoid=False
                         , max_degree=self.max_node_degree, degree_l1=self.degree_l1, degree_l2=self.degree_l2,
                         degree_l3=self.degree_l3

                         , model_size=self.model_size
                         , out_dim_1=self.out_dim_1
                         , out_dim_2=self.out_dim_2)
            if unsupervised and infer:
                mscgnn = mscgnn.classify(MSCGNN_infer=mscgnn, MSCGNN=mscgnn,
                                         embedding_path_name=None  # embedding_name
                                         , embedding_prefix=embedding_name, learning_rate=learning_rate,
                                         aggregator=aggregator, supervised=supervised)

        G = mscgnn.get_graph()
        # if loading then no MSC to equate, does nothing
        mscgnn.equate_graph(G)

        if infer:
            #msc_union.geomsc = mscgnn.geomsc
            #msc_union.arc_dict = mscgnn.arc_dict
            #if load_preprocessed:
            #    G = mscgnn.get_graph()
            msc.equate_graph(G)
            msc_infered = msc
            if union_geomsc:
                geomsc.invert_map()
                msc_infered = geomsc.test_geomsc


        msc_path_and_name = os.path.join(msc_seg_path,
                                         'INFERENCE-Blur' + str(self.blur_sigmas[blur]) + 'pers' + str(
                                             sorted(self.persistence_values)[self.pers_inf_idx]) + '-MSC.tif')
        if draw:
            #mscgnn
            msc_infered.draw_segmentation(filename=msc_path_and_name
                                     , X=inference_image_copy.shape[1], Y=inference_image_copy.shape[2]
                                     , reshape_out=False, dpi=164
                                     , valley=True, ridge=True, original_image=inference_image_copy
                                     , type='predictions')

        infer = False
        if infer:

            inference_mscgnn = mscgnn# MSCGNN()
            inference_image, inference_msc_collection, mask, segmentation = self.train_dataloader[int(self.inference_data_idx)]
            inference_image_copy = copy.deepcopy(inference_image)


            if not load_preprocessed:
                inference_msc = inference_msc_collection[(sorted(self.persistence_values)[self.pers_inf_idx]
                                                          , self.blur_sigmas[blur])]
                inference_mscgnn.assign_msc(msc = inference_msc) #(msc=inference_msc, msc_collection=inference_msc_collection)
                inference_mscgnn.msc_collection = inference_msc_collection

            inference_msc_graph_name = 'inference_msc-feature-graph-' \
                                       + str(self.persistence_values[self.pers_inf_idx]) + 'blur-' + str(self.blur_sigmas[blur])



            inference_msc_graph_name = 'inference_msc-feature-graph' + self.inference_data_idx +'-' \
                                       + str(self.persistence_values[self.pers_inf_idx]) + 'blur-' + str(self.blur_sigmas[blur])

            #
            #   ! need to change subset partition for inference
            #
            if load_preprocessed:
                write_path = self.LocalSetup.drive_training_base_path
                msc_seg_path = os.path.join(write_path, 'msc_seg')

                msc_seg_path = os.path.join(msc_seg_path,
                                            'blur_' + str(self.blur_sigmas[blur])
                                            + 'persistence_' + str(sorted(self.persistence_values)[self.pers_inf_idx]))
                msc_path_and_name = os.path.join(msc_seg_path,
                                                 self.inference_data_idx + 'Blur' + str(self.blur_sigmas[blur]) + 'pers' + str(
                                                     sorted(self.persistence_values)[self.pers_inf_idx]) + '-MSC.tif')

                inference_mscgnn.msc_feature_graph(load_preprocessed=True,
                                                   load_msc=msc_path_and_name,
                                                   image=np.transpose(np.mean(inference_image,axis=1),(1,0)), X=inference_image.shape[0], Y=inference_image.shape[2]
                                            ,persistence_values=self.persistence_values,blur=self.blur_sigmas[blur]
                                         ,val_model='persistence_subset'
                                            , test_samples=0, test_hops=0, accuracy_threshold=0.2
                                            ,write_json_graph_path='./data', name=inference_msc_graph_name
                                            ,test_graph=True)
            else:
                inference_mscgnn.msc_feature_graph(image=np.transpose(np.mean(inference_image, axis=1), (1, 0)),
                                                   X=inference_image.shape[0], Y=inference_image.shape[2]
                                                   , persistence_values=self.persistence_values,
                                                   blur=self.blur_sigmas[blur]
                                                   , val_model='persistence_subset'
                                                   , test_samples=0, test_hops=0, accuracy_threshold=0.2
                                                   , write_json_graph_path='./data', name=inference_msc_graph_name
                                                   , test_graph=True)

            inference_walk_embedding_file = os.path.join(self.LocalSetup.project_base_path, 'datasets', 'walk_embeddings'
                                                         , model_name \
                                                         + 'im_' + str(self.number_images) + '_nWalk-' + str(self.number_walks) +'_walkLength-' + str(self.walk_length)
                                                         +'_pers-' + str(self.persistence_values[self.pers_inf_idx]) + str(self.blur_sigmas[blur])
                                                         + '_idx' + self.inference_data_idx + 'test_walk')

            inference_embedding_name = 'inference_msc-embedding' + self.inference_data_idx +'-pers-' \
                                       + str(self.persistence_values[self.pers_inf_idx]) + 'blur-' + str(self.blur_sigmas[blur])

            # random walks of inference msc
            if not load_preprocessed_walks:
                random_walk_embedding(inference_mscgnn.G, walk_length=self.walk_length
                                      , number_walks=self.number_walks, out_file=inference_walk_embedding_file)

            #
            #  Still need to label all partitions train
            #

            #using trained mscgnn to predict over unlabeled msc
            mscgnn.predict(inference_mscgnn, load_walks=inference_walk_embedding_file, batch_size=int(512/2))

            inference_G = mscgnn.get_graph_prediction()
            inference_mscgnn.equate_graph(inference_G)

            msc_path_and_name = os.path.join(msc_seg_path,
                                             'unseen-predictions-Blur' + str(self.blur_sigmas[blur]) + 'pers' + str(
                                                 sorted(self.persistence_values)[self.pers_inf_idx]) + '-MSC.tif')
            mscgnn.draw_segmentation(image=inference_image_copy
                                     , path_name=msc_path_and_name
                                     , type='predictions')



# Experiment settings
model_name = 'manual_select_from_1st_inference'

other_models = ['manual_select-disjoint_training_wlk-l3-n45_pers1e-8_Feats-polar-nomean',
                'manual_select_training_pers1e-8_polarFeats',
                'manual_select_training_pers1e-7',
                'multi2Pers_3-1_per-1e-4_1e-2_feat[1-3)_im4_wlkn-65-wlkl-3_acc-2.5', #best so far
                'multiPers2_4-2_per-1e-4_1e-2_feat[1-3)_im6_acc-2.5',
                'multiPers3_4-1-2_per-1e-4_1e-3_1e-2_feat[1-3)_im7_acc-2.5',
                'multiPers3_4-2-2_per-1e-4_1e-3_1e-2_feat[1-3)_im8_acc-2.5',
                'multiPers2_6-2_per-1e-3_feat[1-3)_im8_acc-2.5???',
                'multiPers10-4-1_feat[1-4)_im8_acc-2.5',
                  'multiPers6and3_feat[1-4)_im7_acc-2.5']

load_preprocessed = [False, True][0]

load_preprocessed_walks = load_preprocessed

write_msc = not load_preprocessed

collect_features = not load_preprocessed#[False, True][0]

test_param = [False, True][1]

unsupervised = [False, True][0]

mscgnn_learner = MSCGNNTrainer(compute_msc=not load_preprocessed
                               , collect_datasets=True
                               , test_param=test_param)

mscgnn_learner.learn_embedding(val_model='cvt'#persistence_subset'
                , model_name = model_name
                , load_preprocessed=load_preprocessed
                , load_preprocessed_walks=load_preprocessed_walks
                , draw=True, write_msc=write_msc
                , collect_features = collect_features
                , unsupervised=unsupervised            )