import sys
import os
import numpy as np
from skimage import io
import imageio
import argparse
from topoml.MSCGNN import MSCGNN
from topoml.topology.mscnn_segmentation import mscnn_segmentation
from topoml.topology.MSCSegmentation import MSCSegmentation
from topoml.topology.geometric_msc import GeoMSC
from topoml.ml.MSCLearningDataSet import MSCRetinaDataset
from topoml.ui.LocalSetup import LocalSetup
from topoml.ml.MSCSample import MSCSample
from topoml.graphsage.utils import random_walk_embedding

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',default='0', type=str, help="coma separated list of gpus to use")
parser.add_argument('--env', default='multivax',  type=str, help="system to run, if multivax hard assign gpu, sci assumes slurm")
args = parser.parse_args()

LocalSetup = LocalSetup(env=args.env)

pers = 0
blur = 0
persistence_values = [0.001]  # , 0.01, 0.1]#[10, 12, 15, 20 , 23, 25, 30] # below 1 for GeoMSC
blur_sigmas = [1.0]


# Load the Retna data set images and hand segmentations
# (training,test both Stare and Drive), map masks, and reformat images
# to be same size
collect_datasets = True
if collect_datasets:
    print(" %%% collecting data buffers")
    MSCRetinaDataSet = MSCRetinaDataset(with_hand_seg=True)
    # get each set independently for msc computation
    drive_training_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False
                                                                    , drive_training_only=True, env=args.env)
    drive_test_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False, drive_test_only=True, env=args.env)
    drive_test_dataset = MSCRetinaDataset(drive_test_retina_array, split=None, do_transform=False, with_hand_seg=True)
    ##stare_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False, stare_only=True, env=args.env)

    # dataset buffers to use for training:
    drive_training_dataset = MSCRetinaDataset(drive_training_retina_array, split=None, do_transform=False,
                                              with_hand_seg=True)

    # dataset to use for validation
compute_msc = True
drive_training_msc, drive_test_msc, stare_msc = None, None, None
if compute_msc:
    MSCSegmentation = MSCSegmentation()
    print(" %%% computing geometric msc")
    drive_training_dataset = MSCSegmentation.geomsc_segment_images(persistence_values=persistence_values
                                                                   , blur_sigmas=blur_sigmas
                                                                   , data_buffer=drive_training_dataset
                                                                   , data_path=LocalSetup.drive_training_path
                                                                   ,segmentation_path=LocalSetup.drive_training_segmentation_path
                                                                   , write_path=LocalSetup.drive_training_base_path
                                                                   , label=True
                                                                   , save=False
                                                                   , valley=True, ridge=True
                                                                   ,env=args.env)

    drive_test_dataset = MSCSegmentation.geomsc_segment_images(persistence_values = persistence_values, blur_sigmas = blur_sigmas
                                          , data_buffer=drive_test_dataset, data_path = LocalSetup.drive_test_path, segmentation_path=LocalSetup.drive_test_segmentation_path
                                                      ,write_path = LocalSetup.drive_test_base_path, label=True, save=False, valley=True, ridge=True, env=args.env)
    """stare_dataset = MSCSegmentation.geomsc_segment_images(persistence_values = persistence_values, blur_sigmas = blur_sigmas
                                          ,data_buffer=stare_dataset, data_path = LocalSetup.stare_training_data_path, segmentation_path=LocalSetup.stare_segmentations
                                                      ,write_path = LocalSetup.stare_base_path, label=True, save=True, valley=True, ridge=True)"""
    print("%geomsc complete")
## Get total retina array with newly computed msc
## and partition into train, test and val
print(" %%% performing data buffer train, validation, test split ")
drive_training_images, drive_training_msc_collections, drive_training_masks, drive_training_segmentations = list(
    zip(*drive_training_dataset))
drive_test_images, drive_test_msc_collections, drive_test_masks, drive_test_segmentations = list(zip(*drive_test_dataset))
"""stare_images, stare_msc_collections, stare_masks, stare_segmentations = list(zip(*stare_dataset))
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
retina_dataset = drive_test_dataset
MSCRetinaDataSet.retina_array = retina_dataset
#MSCRetinaDataSet.get_retina_array(partial=False, msc=True)
train_dataloader = MSCRetinaDataset(retina_dataset, split="train", do_transform = False, with_hand_seg=True)
val_dataloader = MSCRetinaDataset(retina_dataset, split = "val", do_transform = False, with_hand_seg=True)
test_dataloader = MSCRetinaDataset(retina_dataset, split = "test", do_transform = False, with_hand_seg=True)


print(" %%% data buffer split complete")

inference_image, msc, mask, segmentation = train_dataloader[0]
inference_msc_graph_name = 'inference_msc-feature-graph-' + str(persistence_values[pers]) + 'blur-' + str(blur)
inference_msc = msc[(persistence_values[pers], blur_sigmas[blur])]

def GeoMSC_Inference(mscgnn, inference_msc, inference_image,
                     embedding_name, learning_rate, aggregator
                     , persistence, blur, trained_prefix=None, gpu=0, env=None):


    # Can also pass graph if contained in gnn
    inference_mscgnn = MSCGNN(msc=inference_msc)

    inference_msc_graph_name = 'inference_msc-feature-graph-' + str(persistence) + 'blur-' + str(blur)
    inference_mscgnn.msc_feature_graph(image=np.transpose(np.mean(inference_image, axis=1), (1, 0)), X=inference_image.shape[0], Y=inference_image.shape[2]
                             , validation_samples=1, validation_hops=20
                             , test_samples=1, test_hops=20, accuracy_threshold=0.1
                         , write_json_graph_path='./data', name=inference_msc_graph_name)

    walk_embedding_file = os.path.join(LocalSetup.project_base_path, 'datasets', 'walk_embeddings'
                                       , str(persistence) + str(blur) + 'test_walk')
    inference_embedding_name = 'inference_msc-embedding-pers-'+str(persistence)+'blur-'+str(blur)

    embedding_path_name = None#embedding_name + '-unsup-json_graphs' + '/' + aggregator + '_' + 'big'
    #embedding_path_name += ("_{lr:0.6f}").format(lr=learning_rate)
    #mscgnn.embed_inference_msc(inference_mscgnn=inference_mscgnn,persistence=persistence,blur=blur
    #                           ,inference_embedding_file=inference_embedding_name, embedding_name=embedding_name
    #inference_mscgnn = mscgnn

    # if embedding graph made with test/train set the same (and named the same)
    if trained_prefix is not None:
        inference_mscgnn = mscgnn.classify(embedding_prefix=embedding_name,MSCGNN_infer=inference_mscgnn,
                         aggregator=aggregator, embedding_path_name=embedding_path_name
                        ,trained_prefix=trained_prefix, learning_rate=learning_rate)
    else:
        # adjust classification to use mscgnn for inference with known
        # gnn and get new inference mscgnn embedding.
        inference_mscgnn = mscgnn.classify(MSCGNN_infer=inference_mscgnn, MSCGNN=mscgnn, embedding_path_name=embedding_path_name
                        , embedding_prefix=embedding_name, learning_rate=learning_rate, aggregator=aggregator)


    # Hand select test graph
    """gnn.select_msc( train_or_test='test', group_name= train_group_name,bounds=[[451,700],[301,600]], write_json=True, write_msc=True, load_msc=False)
    """

    # if loading graph (test/train) for classification
    ###gnn.classify(test_prefix=train_group_name, trained_prefix=train_group_name, embedding_prefix=embedding_name,
    ###            aggregator=aggregator, learning_rate=learning_rate)


    # Show labaling assigned by trained model
    inference_msc = inference_mscgnn.geomsc
    inference_msc.draw_segmentation(filename="test.tiff"
                                    ,original_image=inference_image
                                    ,X=inference_image.shape[0],Y=inference_image.shape[2]
                                    ,reshape_out=False ,dpi = 50
                                    , valley=True, ridge=True)
    ##mscgnn.show_gnn_classification(pred_graph_prefix=embedding_name, train_view=False)  # embedding_name)

    # see train and val sets, must put in directory log-dir and make new folder
    # with appropriate name of train graph, e.g. looks for graph in log-dir
    ###mscgnn.show_gnn_classification(pred_graph_prefix=train_group_name, train_view=True)

retina_dataset_train = drive_training_dataset
MSCRetinaDataSet.retina_array = retina_dataset_train
#MSCRetinaDataSet.get_retina_array(partial=False, msc=True)
train_dataloader_trained = MSCRetinaDataset(retina_dataset_train, split="train", do_transform = False, with_hand_seg=True)
val_dataloader_trained = MSCRetinaDataset(retina_dataset_train, split = "val", do_transform = False, with_hand_seg=True)
test_dataloader_trained = MSCRetinaDataset(retina_dataset_train, split = "test", do_transform = False, with_hand_seg=True)
pers = 0
blur = 0
image, msc, mask, segmentation = train_dataloader[0]
msc = msc[(persistence_values[pers], blur_sigmas[blur])]
mscgnn = MSCGNN(msc=msc)

trained_msc_graph_name =  'msc-feature-graph-' + str(persistence_values[pers]) + 'blur-' + str(blur)
mscgnn.msc_feature_graph(image=np.transpose(np.mean(image, axis=1), (1, 0)), X=image.shape[0], Y=image.shape[2]
                         , validation_samples=1, validation_hops=20
                         , test_samples=1, test_hops=20, accuracy_threshold=0.1
                         , write_json_graph_path='./data', name=trained_msc_graph_name)

aggregator = ['graphsage_maxpool', 'gcn', 'graphsage_seq','graphsage_maxpool', 'graphsage_meanpool','graphsage_seq', 'n2v'][4]
learning_rate = 0.25

GeoMSC_Inference(inference_msc=inference_msc
                 , mscgnn=mscgnn
                 , inference_image=inference_image
                 ,aggregator=aggregator
                 ,persistence=persistence_values[pers]
                 ,blur=blur_sigmas[blur]
                 ,learning_rate=0.25
                 ,embedding_name='msc-embedding-pers-0.001blur-0'#-unsup-json_graphs/graphsage_meanpool_big_0.250000'
                 ,gpu=args.gpu
                 ,env=args.env)