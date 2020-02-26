import sys
import os
from skimage import io
import imageio

from topoml.topology.mscnn_segmentation import mscnn_segmentation
from topoml.topology.MSCSegmentation import MSCSegmentation
from topoml.topology.geometric_msc import GeoMSC
from topoml.ml.MSCLearningDataSet import MSCRetinaDataset
from topoml.ui.LocalSetup import LocalSetup
from topoml.ml.MSCSample import MSCSample

#sys.path.append(os.getcwd())



paths_for_multivax=""" training_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/drive/DRIVE/training/images"
#training_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/neuron_msc"
testing_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/drive/DRIVE/test/images"
train_write_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/drive/DRIVE/training/" # "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/" #
test_write_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/drive/DRIVE/test/"
stare_training_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/stare/images"
stare_train_write_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/stare/"
"""
# Paths for Multivax
project_base_path = "/home/sam/Documents/PhD/Research/GeoMSCxML/"
training_data_path = os.path.join(project_base_path,"datasets","optics","drive","DRIVE","training","images")
training_seg_data_path = os.path.join(project_base_path,"datasets","optics","drive","DRIVE","training","1st_manual")
#training_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/neuron_msc"
testing_data_path = os.path.join(project_base_path,"datasets","optics","drive","DRIVE","test","images")
train_write_path = os.path.join(project_base_path,"datasets","optics","drive","DRIVE","training")
# "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/" #
test_write_path = os.path.join(project_base_path,"datasets","optics","drive","DRIVE","test")
stare_training_data_path = os.path.join(project_base_path,"datasets","optics","stare","images")
stare_train_write_path = os.path.join(project_base_path,"datasets","optics","stare")

# declarations for running examples
persistence_values = [0.001]#, 0.01, 0.1]#[10, 12, 15, 20 , 23, 25, 30] # below 1 for GeoMSC
blur_sigmas = [1.0]#1.0, 2.0, 5.0, 10] # add iterative blur &  try multichannel
#traininig_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/neuron_msc"
#train_write_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/"

# run the MSC segmentation over images
"""msc_segment_images(persistence_values = persistence_values, blur_sigma = blur_sigma,
                   data_path = training_data_path, write_path = train_write_path)"""

#msc_segment_images(persistence_values = persistence_values, blur_sigma = blur_sigma,
#                   data_path = stare_training_data_path, write_path = train_write_path)

#msc_segment_images(persistence_values = persistence_values, blur_sigma = blur_sigma,
#                   data_path = testing_data_path, write_path = test_write_path)

# Local Paths
LocalSetup = LocalSetup()

# Load the Retna data set images and hand segmentations
# (training,test both Stare and Drive), map masks, and reformat images
# to be same size
collect_datasets = True
if collect_datasets:
    MSCRetinaDataSet = MSCRetinaDataset(with_hand_seg=True)
    # get each set independently for msc computation
    drive_training_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False, drive_training_only=True)
    ##drive_test_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False, drive_test_only=True)
    ##stare_retina_array = MSCRetinaDataSet.get_retina_array(partial=False, msc=False, stare_only=True)

    #dataset buffers to use for training:
    drive_training_dataset = MSCRetinaDataset(drive_training_retina_array, split=None, do_transform = False, with_hand_seg=True)
    #dataset to use for validation
    ##drive_test_dataset = MSCRetinaDataset(drive_test_retina_array, split = None, do_transform = False, with_hand_seg=True)
    #dataset to use for testing
    ##stare_dataset = MSCRetinaDataset(stare_retina_array, split = None,do_transform = False, with_hand_seg=True)

# run the Geometric MSC segmentation over images
compute_msc = True
drive_training_msc, drive_test_msc, stare_msc = None, None, None
if compute_msc:
    MSCSegmentation = MSCSegmentation()
    drive_training_dataset = MSCSegmentation.geomsc_segment_images(persistence_values = persistence_values
                                                               , blur_sigmas = blur_sigmas
                                                               ,data_buffer=drive_training_dataset
                                                               , data_path = LocalSetup.drive_training_path
                                                               , segmentation_path=LocalSetup.drive_training_segmentation_path
                                                               ,write_path = LocalSetup.drive_training_base_path
                                                               , label=True
                                                               , save=True
                                                               , valley=True, ridge=True)

    """drive_test_dataset = MSCSegmentation.geomsc_segment_images(persistence_values = persistence_values, blur_sigmas = blur_sigmas
                                          , data_buffer=drive_test_dataset, data_path = LocalSetup.drive_test_path, segmentation_path=LocalSetup.drive_test_segmentation_path
                                                      ,write_path = LocalSetup.drive_test_base_path, label=True, save=True, valley=True, ridge=True)
    stare_dataset = MSCSegmentation.geomsc_segment_images(persistence_values = persistence_values, blur_sigmas = blur_sigmas
                                          ,data_buffer=stare_dataset, data_path = LocalSetup.stare_training_data_path, segmentation_path=LocalSetup.stare_segmentations
                                                      ,write_path = LocalSetup.stare_base_path, label=True, save=True, valley=True, ridge=True)"""
## Get total retina array with newly computed msc
## and partition into train, test and val
drive_training_images, drive_training_msc_collections, drive_training_masks, drive_training_segmentations = list(zip(*drive_training_dataset))

print("number drive training msc (number pers*buffer): ", len(drive_training_msc_collections))
print("number per pers,buffer: ", len(drive_training_msc_collections[0]))
print(" image shape: ",drive_training_images[0].shape)
msc = drive_training_msc_collections[0][(persistence_values[0],blur_sigmas[0])]
msc = MSCSample(geomsc=msc)
val_positive_subset, val_negative_subset = msc.sample_graph_neighborhoods(X=drive_training_images[0].shape[1]
                                                        ,Y=drive_training_images[0].shape[2]
                                                        ,accuracy_threshold=0.1,count=20, rings=10, seed=66)
print("neg ",len(val_negative_subset))
print("pos ", len(val_positive_subset))
#drive_test_images, drive_test_msc_collections, drive_test_masks, drive_test_segmentations = list(zip(*drive_test_dataset))
#stare_images, stare_msc_collections, stare_masks, stare_segmentations = list(zip(*stare_dataset))
retina_dataset = list(zip(drive_training_images #+ drive_test_images + stare_images
                          ,drive_training_msc_collections #+ drive_test_msc_collections + stare_msc_collections
                          ,drive_training_masks# + drive_test_masks #+ stare_masks
                          ,drive_training_segmentations ))#+ drive_test_segmentations ))#+ stare_segmentations))
                          
MSCRetinaDataSet.retina_array = retina_dataset
#MSCRetinaDataSet.get_retina_array(partial=False, msc=True)
train_dataloader = MSCRetinaDataset(retina_dataset, split="train", do_transform = False, with_hand_seg=True)
val_dataloader = MSCRetinaDataset(retina_dataset, split = "val", do_transform = False, with_hand_seg=True)
test_dataloader = MSCRetinaDataset(retina_dataset, split = "test", do_transform = False, with_hand_seg=True)

# sanity check to view msc
msc = GeoMSC()

fname_base="/home/sam/Documents/PhD/Research/GeoMSCxML/datasets/optics/drive/DRIVE/training/msc_seg/persistence_0.01/33_training_pers-0.01-MSC"
msc.read_from_file(fname_base,labeled=True)
msc.draw_segmentation(filename=fname_base+"test.tiff"
                      ,X=304,Y=352
                      ,reshape_out=False ,dpi = 50
                      , valley=True, ridge=False)