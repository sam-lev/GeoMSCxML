import sys
import os
from skimage import io
import imageio
import numpy as np
import copy

from topoml.topology.mscnn_segmentation import mscnn_segmentation
from topoml.ui.ArcSelector import ArcSelector
from topoml.ml.MSCSample import MSCSample
#class with local file paths
from topoml.ui.LocalSetup import LocalSetup as LS

sys.path.append(os.getcwd())


LocalSetup = LS()
paths_for_multivax = LocalSetup.paths_for_multivax
# Paths for Multivax
project_base_path = LocalSetup.project_base_path
training_data_path = LocalSetup.drive_training_path
training_seg_data_path  = LocalSetup.drive_training_segmentation_path
testing_data_path = LocalSetup.drive_test_path
train_write_path = LocalSetup.drive_training_path
test_write_path = LocalSetup.test_write_path
stare_training_data_path = LocalSetup.stare_training_data_path
#stare_train_write_path = LocalSetup.stare_train_write_path

class MSCSegmentation:

    def __init__(self, msc=None, geomsc=None, labeled_segmentation=None, ridge=True, valley=False):
        self.msc = [] if msc is None else [msc]
        self.geomsc = [] if geomsc is None else [geomsc]
        self.labeled_segmentation = labeled_segmentation
        # msc after labeling arcs (updating arc.label_accuracy
        self.labeled_msc = []


    # use MSCTrainingSet to label msc segmentation for supervised learning
    # applying labels from hand segmented ground truth images.
    def label_msc(self, msc=None, geomsc=None, labeled_segmentation=None, labeled_mask=None, invert=False):
        supervised_MSC = MSCSample(msc=msc, geomsc=geomsc, labeled_segmentation=labeled_segmentation)
        labeled_msc = supervised_MSC.map_labeling(msc=msc, geomsc=geomsc, labeled_segmentation=labeled_segmentation,
                                          labeled_mask = labeled_mask, invert=invert)
        return labeled_msc


    # take raw 'image' used to compute msc and draw
    # msc valley and/or ridge edges displayed as
    # binary map if no labeling or color map if labeled based
    # off of ground truth hand segmentation determined by
    # percent overlap with labeled segmentation
    def draw_msc(self, image=None, image_filename=None, save_name=None, write_path='.'
                 ,msc=None, valley=True, ridge=False, invert=True, persistence=0):
        # check needed folders present else make
        p = persistence
        if image_filename is not None:
            image = io.imread(image_filename)#, as_gray=False)
            if save_name is None:
                save_name = image_filename.rsplit('/', 1)[1].rsplit('.', 1)[0]

        # ground truth segmentation
        raw_path = os.path.join(write_path, 'raw_images')
        # write folder path
        if not os.path.exists(os.path.join(project_base_path, train_write_path,'labeled_msc')):
            os.mkdir(os.path.join(project_base_path, train_write_path,'labeled_msc'))
        msc_seg_path = os.path.join(project_base_path, train_write_path,'labeled_msc')
        # persistence specific folder path
        if not os.path.exists(os.path.join(msc_seg_path, 'persistence_' + str(persistence))):
            os.mkdir(os.path.join(msc_seg_path, 'persistence_' + str(persistence)))
        msc_seg_path = os.path.join(msc_seg_path, 'persistence_' + str(persistence))

        # path name of msc to save to
        save_path_name = os.path.join(msc_seg_path, save_name + '_pers-' + str(persistence) + '-MSC.tif')

        msc_selector = ArcSelector(image=image, msc=msc, valley=True, ridge=True, invert=True)
        unsup_seg_in_arcs, unsup_seg_out_arcs, unsup_seg_pixels = msc_selector.draw_segmentation(save_path_name,
                                                                                                        msc=msc,
                                                                                                        invert=True)



    # iterate over various persistence values and compute the
    # morse smale segmentations for a number of images located at data_path
    # will create a folder at write path with raw images and another
    # fodler with msc segmentations for wach image at each persistence
    def msc_segment_images(self, persistence_values = [1], blur_sigmas = [3], data_path = '.'
                           , write_path = '.'):
        # check needed folders present else make
        if not os.path.exists(os.path.join(write_path, 'raw_images')):
            os.mkdir(os.path.join(write_path, 'raw_images'))
        # iterate through images and compute msc for each image
        # at various persistence values
        images = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and any(image_type in f.rsplit('.', 1)[1] for image_type in ['tif','gif','jpg','png','ppm'])]

        for img in images:
            for blur_sigma in blur_sigmas:
                for pers in persistence_values:
                    # construct msc object
                    mscnn = mscnn_segmentation()
                    mscnn.clear_msc()
                    msc = mscnn.compute_msc(image =  os.path.join(data_path,img), persistence = pers,
                                            blur_sigma=blur_sigma, write_path = write_path)
                    mscnn.construct_msc_from_image(image = os.path.join(data_path,img),
                                                   write_path = write_path, persistence = pers,
                                                   blur_sigma = blur_sigma)
                    self.msc.append(msc)


    # uses mscnn_segmentation class to construct geometric MSC
    # over given images and given persistences. Geometric in that
    # vertices added at edge intersections including ridge lines
    # and not only , min/max
    def geomsc_segment_images(self, persistence_values = [1], blur_sigmas = [3]
                              , data_buffer = None, data_path = None, segmentation_path=None,
                              write_path = None, labeled_segmentation=None, label=True
                              , save=False, save_binary_seg=False, number_images=None, persistence_cardinality = None
                              , valley=True, ridge=True, env='multivax'):
        LocalSetup = LS(env=env)


        # check needed folders present else make
        if not os.path.exists(os.path.join(write_path, 'raw_images')):
            os.mkdir(os.path.join(write_path, 'raw_images'))
        # iterate through images and compute msc for each image
        # at various persistence values
        images = None
        if data_path and segmentation_path is not None:
            images = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and any(image_type in f.rsplit('.', 1)[1] for image_type in ['tif','gif','jpg','png','ppm','vk.ppm'])])
            seg_image_paths = sorted([f for f in os.listdir(segmentation_path)])
            seg_images = []
            for path in seg_image_paths:
                #print(segmentation_path," path: ", path)
                seg_images.append(imageio.imread(os.path.join(segmentation_path, path)))
            im_count = range(len(images))
            if data_buffer is None:
                print("no data buffer given composing new data buffer")
                data_buffer = zip(images, seg_images, im_count)

        if persistence_cardinality is None:
            persistence_cardinality = {}
            for i in range(number_images):
                persistence_cardinality[i] = number_images
        persistence_cardinality = copy.deepcopy(persistence_cardinality)

        msc_segmentations = []
        images_ = []
        masks_ = []
        segmentations_ = []
        count = 0 #if images is None else len(images)-1
        #images=images[:number_images]
        image_cap = -1
        for image, msc_collection , mask, segmentation in data_buffer:
            image_cap+=1
            if number_images is not None and image_cap == number_images:
                break

            #if image.shape[0] <= 3:
            #    image  = np.transpose(image,[1,2,0])
            #if mask.shape[0] <= 3:
            #    mask = np.transpose(mask, [1, 2, 0])
            #if segmentation.shape[0] <= 3:
            #    segmentation = np.transpose(segmentation, [1, 2, 0])
            im_path = images[count] #if images is not None else ""
            labeled_segmentation = None
            labeled_mask = None
            if segmentation is not None:
                # labeled_segmentation= np.zeros([0, 1, 304, 352])
                segmentation[segmentation > 0] = 1
                #labeled_segmentation = segmentation * mask if type(mask) is not int else segmentation
                labeled_segmentation = np.mean(segmentation, axis=0)
            if mask is not None:
                # labeled_segmentation= np.zeros([0, 1, 304, 352])
                mask[mask > 0] = 1
                # labeled_segmentation = segmentation * mask if type(mask) is not int else segmentation
                labeled_mask = np.mean(mask, axis=0)
            # collect to return data buffer with msc
            images_.append(image)
            masks_.append(mask)
            segmentations_.append(segmentation)
            count+=1
            msc_collection= {}
            for blur_sigma in sorted(blur_sigmas):
                for pers_count, pers in enumerate(sorted(persistence_values)):
                    pers_cap = persistence_cardinality[pers_count]
                    if pers_cap <= 0:
                        continue
                    # construct msc object
                    mscnn = mscnn_segmentation()
                    mscnn.clear_msc()
                    #compute geometric msc
                    mscnn.image = copy.deepcopy(image)
                    image_name_and_path = os.path.join(data_path,im_path)
                    print(">>>>")
                    print(image_name_and_path)
                    print(">>>>")
                    msc = mscnn.compute_geomsc(image_filename =  image_name_and_path
                                               ,image=mscnn.image
                                               ,X=image.shape[2], Y=image.shape[1]
                                               ,geomsc_exec_path=os.path.join(LocalSetup.project_base_path,'..')
                                               , persistence = pers
                                               , blur=True
                                               , blur_sigma=blur_sigma
                                               , grey_scale=True
                                               , write_path = write_path
                                               , scale_intensities=False
                                               , augment_channels = [])

                    mscnn.msc= msc
                    if label:
                        labeled_msc = self.label_msc(geomsc=msc
                                                     ,labeled_segmentation=labeled_segmentation
                                                     ,labeled_mask=labeled_mask,invert=True)
                        mscnn.msc = labeled_msc
                        mscnn.geomsc = labeled_msc
                        msc = labeled_msc
                    # compute geomsc over image
                    if save:
                        image_filename =  os.path.join(data_path,im_path)
                        img_name = image_filename.rsplit('/', 1)[1].rsplit('.', 1)[0]
                        msc_seg_path = os.path.join(write_path, 'msc_seg')

                        if not os.path.exists(os.path.join(msc_seg_path, 'blur_'+str(blur_sigma)+'persistence_' + str(pers))):
                            os.mkdir(os.path.join(msc_seg_path,'blur_'+str(blur_sigma)+ 'persistence_' + str(pers)))
                        msc_seg_path = os.path.join(msc_seg_path, 'blur_'+str(blur_sigma)+ 'persistence_' + str(pers))



                        seg_img = os.path.join(write_path, 'ground_truth_seg', img_name + '_seg.gif')
                        msc_path_and_name = os.path.join(msc_seg_path, str(count-1) + 'Blur'+str(blur_sigma)+'pers' + str(pers) + '-MSC.tif')
                        image_copy = copy.deepcopy(image)
                        msc.draw_segmentation(filename=msc_path_and_name
                                              , X=image_copy.shape[1], Y=image_copy.shape[2]
                                              , reshape_out=False, dpi=164
                                              , valley=True, ridge=True,original_image=mscnn.image)
                        #mscnn.construct_geomsc_from_image(image_filename = os.path.join(data_path,im_path)
                        #                                  ,image=image
                        #                                  , write_path = write_path
                        #                                  , persistence = pers
                        #                                  , blur_sigma = blur_sigma
                        #                                  ,binary=save_binary_seg
                        #                                  ,valley=valley,ridge=ridge)


                        msc.write_msc(filename=msc_path_and_name, msc=msc, label=label)

                    persistence_cardinality[pers_count] = pers_cap - 1
                    print("computed msc for persistence:")
                    print(pers)
                    print("over image:")
                    print(image_name_and_path)

                    msc_collection[(pers,blur_sigma)] = msc
                    self.msc.append(msc)

            msc_segmentations.append(msc_collection)

        if data_buffer is not None:
            data_buffer_with_msc = list(zip(images_
                                            ,msc_segmentations
                                            ,masks_
                                            ,segmentations_))
            return data_buffer_with_msc
        else:
            return msc_segmentations




                #draw_msc(image = None, image_filename = os.path.join(data_path,img), save_name=None,
                #         write_path = '.',msc = labeled_msc, valley = True, ridge = False,
                #         invert=True, persistence=pers)




    # declarations for running examples
    """persistence_values = [0.01]#[10, 12, 15, 20 , 23, 25, 30] # below 1 for GeoMSC
    blur_sigmas = [2.0] """
    #traininig_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/neuron_msc"
    #train_write_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/"

    # run the MSC segmentation over images
    """msc_segment_images(persistence_values = persistence_values, blur_sigma = blur_sigma,
                       data_path = training_data_path, write_path = train_write_path)"""

    #msc_segment_images(persistence_values = persistence_values, blur_sigma = blur_sigma,
    #                   data_path = stare_training_data_path, write_path = train_write_path)

    #msc_segment_images(persistence_values = persistence_values, blur_sigma = blur_sigma,
    #                   data_path = testing_data_path, write_path = test_write_path)

    # run the Geometric MSC segmentation over images
    """geomsc_segment_images(persistence_values = persistence_values
                          , blur_sigma = blur_sigmas
                          , data_path = training_data_path
                          , write_path = train_write_path)"""
