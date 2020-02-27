
import sys
import os
import numpy as np
from skimage import io
import imageio
from topoml.MSCGNN import MSCGNN
from topoml.topology.mscnn_segmentation import mscnn_segmentation
from topoml.topology.MSCSegmentation import MSCSegmentation
from topoml.topology.geometric_msc import GeoMSC
from topoml.ml.MSCLearningDataSet import MSCRetinaDataset
from topoml.ui.LocalSetup import LocalSetup
from topoml.ml.MSCSample import MSCSample
from topoml.graphsage.utils import random_walk_embedding

# sys.path.append(os.getcwd())


##
##
##
##
##


# declarations for running examples
persistence_values = [0.001]  # , 0.01, 0.1]#[10, 12, 15, 20 , 23, 25, 30] # below 1 for GeoMSC
blur_sigmas = [1.0]  # 1.0, 2.0, 5.0, 10] # add iterative blur &  try multichannel
# traininig_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/neuron_msc"
# train_write_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/"

# run the MSC segmentation over images
"""msc_segment_images(persistence_values = persistence_values, blur_sigma = blur_sigma,
                   data_path = training_data_path, write_path = train_write_path)"""

# msc_segment_images(persistence_values = persistence_values, blur_sigma = blur_sigma,
#                   data_path = stare_training_data_path, write_path = train_write_path)

# msc_segment_images(persistence_values = persistence_values, blur_sigma = blur_sigma,
#                   data_path = testing_data_path, write_path = test_write_path)

# Local Paths
LocalSetup = LocalSetup(env='sci')

# Load the Retna data set images and hand segmentations
# (training,test both Stare and Drive), map masks, and reformat images
# to be same size
collect_datasets = True
if collect_datasets:
    print(" %%% collecting data buffers")
    MSCRetinaDataSet = MSCRetinaDataset(with_hand_seg=True)
    # get each set independently for msc computation
    drive_training_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False
                                                                    , drive_training_only=True, env='sci')
    ##drive_test_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False, drive_test_only=True)
    ##stare_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False, stare_only=True)

    # dataset buffers to use for training:
    drive_training_dataset = MSCRetinaDataset(drive_training_retina_array, split=None, do_transform=False,
                                              with_hand_seg=True)
    # dataset to use for validation
    ##drive_test_dataset = MSCRetinaDataset(drive_test_retina_array, split = None, do_transform = False, with_hand_seg=True)
    # dataset to use for testing
    ##stare_dataset = MSCRetinaDataset(stare_retina_array, split = None,do_transform = False, with_hand_seg=True)
    print("%collection complete")
# run the Geometric MSC segmentation over images
compute_msc = True
drive_training_msc, drive_test_msc, stare_msc = None, None, None
if compute_msc:
    MSCSegmentation = MSCSegmentation()
    print(" %%% computing geometric msc")
    drive_training_dataset = MSCSegmentation.geomsc_segment_images(persistence_values=persistence_values
                                                                   , blur_sigmas=blur_sigmas
                                                                   , data_buffer=drive_training_dataset
                                                                   , data_path=LocalSetup.drive_training_path
                                                                   ,
                                                                   segmentation_path=LocalSetup.drive_training_segmentation_path
                                                                   , write_path=LocalSetup.drive_training_base_path
                                                                   , label=True
                                                                   , save=True
                                                                   , valley=True, ridge=True)

    """drive_test_dataset = MSCSegmentation.geomsc_segment_images(persistence_values = persistence_values, blur_sigmas = blur_sigmas
                                          , data_buffer=drive_test_dataset, data_path = LocalSetup.drive_test_path, segmentation_path=LocalSetup.drive_test_segmentation_path
                                                      ,write_path = LocalSetup.drive_test_base_path, label=True, save=True, valley=True, ridge=True)
    stare_dataset = MSCSegmentation.geomsc_segment_images(persistence_values = persistence_values, blur_sigmas = blur_sigmas
                                          ,data_buffer=stare_dataset, data_path = LocalSetup.stare_training_data_path, segmentation_path=LocalSetup.stare_segmentations
                                                      ,write_path = LocalSetup.stare_base_path, label=True, save=True, valley=True, ridge=True)"""
    print("%geomsc complete")
## Get total retina array with newly computed msc
## and partition into train, test and val
print(" %%% performing data buffer train, validation, test split ")
drive_training_images, drive_training_msc_collections, drive_training_masks, drive_training_segmentations = list(
    zip(*drive_training_dataset))
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
retina_dataset = drive_training_dataset
MSCRetinaDataSet.retina_array = retina_dataset
#MSCRetinaDataSet.get_retina_array(partial=False, msc=True)
train_dataloader = MSCRetinaDataset(retina_dataset, split="train", do_transform = False, with_hand_seg=True)
val_dataloader = MSCRetinaDataset(retina_dataset, split = "val", do_transform = False, with_hand_seg=True)
test_dataloader = MSCRetinaDataset(retina_dataset, split = "test", do_transform = False, with_hand_seg=True)
print(" %%% data buffer split complete")

def learn_embedding():

    print(" %%%%% creating geomsc feature graph ")

    # for image, msc, mask, segmentation in train_dataloader:
    pers = 0
    blur = 0
    image, msc, mask, segmentation = train_dataloader[0]
    msc = msc[(persistence_values[pers], blur_sigmas[blur])]
    mscgnn = MSCGNN(msc=msc)
    mscgnn.msc_feature_graph(image=np.transpose(np.mean(image,axis=1),(1,0)), X=image.shape[0], Y=image.shape[2]
                     ,validation_samples=1, validation_hops=20
                     , test_samples=1, test_hops=20, accuracy_threshold=0.1)

    print(" %%%%% feature graph complete")

    # construct unsupervised gnn model
    aggregator = ['graphsage_maxpool', 'gcn', 'graphsage_seq','graphsage_maxpool'
        , 'graphsage_meanpool','graphsage_seq', 'n2v'][4]

    print('... Beginning training with aggregator: ', aggregator)

    # Random walks used to determine node pairs for unsupervised loss.
    # to make a random walk collection use ./topoml/graphsage/utils.py
    # example run: python3 topoml/graphsage/utils.py ./data/json_graphs/test_ridge_arcs-G.json ./data/random_walks/full_msc_n-1_k-40
    ##load_walks='full_msc_n-1_k-40'
    ##load_walks='full_msc_n-1_k-50_Dp-10_nW-100'
    print('... Generating Random Walk Neighborhoods for Node Co-Occurance')
    walk_embedding_file = os.path.join(LocalSetup.project_base_path,'datasets','walk_embeddings'
                                       ,str(persistence_values[0])+str(blur_sigmas[0]) +'test_walk')
    random_walk_embedding(mscgnn.G, walk_length=10, number_walks=20, out_file=walk_embedding_file)

    mscgnn.unsupervised(aggregator=aggregator, slurm=False)

    # hyper-param for gnn
    learning_rate = 2e-8
    polarity = 25
    weight_decay = 0.1
    epochs = 10
    depth = 3

    # file name for embedding of trained model

    embedding_name = 'msc-embedding-pers-'+str(persistence_values[pers])+'blur-'+str(blur)
    #'n-' + str(cvt[0]) + '_k-' + str(cvt[1]) + '_lr-' + str(learning_rate) + 'Plrty-' + str(
       #polarity) + '_epochs-' + str(epochs) + '_depth-' + str(depth) + 'trainGrp-' + train_group_name

    mscgnn.train( embedding_name = embedding_name, load_walks=walk_embedding_file
                  , learning_rate=learning_rate, epochs=epochs
                  , weight_decay=weight_decay, polarity=polarity, depth=depth)


def GeoMSC_Inference(mscgnn, inference_msc, inference_image,
                     embedding_name, learning_rate, aggregator, persistence, blur):

    # if embedding graph made with test/train set the same (and named the same)
    if embedding_name is not None:
        mscgnn.classify( embedding_prefix=embedding_name, aggregator = aggregator
                     , learning_rate=learning_rate)
    else:
        # Can also pass graph if contained in gnn
        inference_mscgnn = MSCGNN(msc=inference_msc)

        inference_mscgnn.msc_feature_graph(image=np.transpose(np.mean(inference_image, axis=1), (1, 0)), X=inference_image.shape[0], Y=inference_image.shape[2]
                                 , validation_samples=1, validation_hops=20
                                 , test_samples=1, test_hops=20, accuracy_threshold=0.1)

        walk_embedding_file = os.path.join(LocalSetup.project_base_path, 'datasets', 'walk_embeddings'
                                           , str(persistence) + str(blur) + 'test_walk')
        inference_embedding_name = 'msc-embedding-pers-'+str(persistence)+'blur-'+str(blur)

        mscgnn.embed_inference_msc(inference_mscgnn=inference_mscgnn,persistence=persistence,blur=blur,inference_embedding_file=inference_embedding_name, walk_embedding_file=walk_embedding_file)
        # adjust classification to use mscgnn for inference with known
        # gnn and get new inference mscgnn embedding.
        mscgnn.classify(MSCGNN_infer=inference_mscgnn)


    # Hand select test graph
    """gnn.select_msc( train_or_test='test', group_name= train_group_name,bounds=[[451,700],[301,600]], write_json=True, write_msc=True, load_msc=False)
    """

    # if loading graph (test/train) for classification
    ###gnn.classify(test_prefix=train_group_name, trained_prefix=train_group_name, embedding_prefix=embedding_name,
    ###            aggregator=aggregator, learning_rate=learning_rate)


    # Show labaling assigned by trained model
    mscgnn.show_gnn_classification(pred_graph_prefix=embedding_name, train_view=False)  # embedding_name)

    # see train and val sets, must put in directory log-dir and make new folder
    # with appropriate name of train graph, e.g. looks for graph in log-dir
    ###mscgnn.show_gnn_classification(pred_graph_prefix=train_group_name, train_view=True)


learn_embedding()