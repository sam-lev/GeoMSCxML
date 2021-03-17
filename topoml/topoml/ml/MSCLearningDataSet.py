
#external imports
import os
import gzip
import shutil
import tarfile
import zipfile
import imageio
from skimage import io
import rawpy
#from torch.utils.data import Dataset
import numpy as np
#internal imports
from topoml.topology.MSCSegmentation import MSCSegmentation
from topoml.ml.utils import get_split
from topoml.image.utils import resize_img
from topoml.image.utils import remove_small_regions
from topoml.ui.LocalSetup import LocalSetup


class MSCLearningDataSet:
        def __init__(self, persistence_values=[], blur_sigmas=[], training_data_path=None, validation_data_path=None, test_data_path=None
                    ,training_write_path = None, validation_write_path = None, test_write_path = None
                     , training_set=None, validation_set=None, test_set=None):

            self.validation_set = validation_set
            self.training_set = training_set
            self.test_set = test_set
            self.data_array = []


        def __getitem__(self, index):
            sample = [x for x in self.data_array[index]]  # torch.FloatTensor(x)
            """if self.do_transform:
                v_gen = RandomVerticalFlipGenerator()
                h_gen = RandomHorizontalFlipGenerator()
                t = Compose([
                    v_gen,
                    RandomVerticalFlip(gen=v_gen),
                    h_gen,
                    RandomHorizontalFlip(gen=h_gen),
                ])
                sample = t(sample)"""
            return sample

        def __len__(self):
            return len(self.data_array)

        def read_images(self, filetype, dest_folder, color_invert=False, dim_invert = False):
            # zip_ref = zipfile.ZipFile('DRIVE.zip', 'r')
            # zip_ref.extractall('datasets/drive')
            # zip_ref.close()
            if filetype != 'raw':
                all_images = []
                sorted_items = sorted(os.listdir(dest_folder))
                for item in sorted_items:
                    if dest_folder[-1] != '/':
                       dest_folder = dest_folder+"/"
                    if item.endswith(filetype):
                        img = imageio.imread(dest_folder + item)
                        if len(img.shape) == 3:
                            img = np.pad(img, ((12, 12), (69, 70), (0, 0)), mode='constant')
                        else:
                            img = np.pad(img, ((12, 12), (69, 70)), mode='constant')
                        #img = resize_img(img)
                        img = img / 255.
                        img = img.astype(np.float32)
                        if len(img.shape) == 2:
                            img = img.astype(np.float32)
                            img = np.expand_dims(img, axis=2)
                        all_images.append(img)
                        #print("drive read im shape ", img.shape)
            else:
                all_images = []
                sorted_items = sorted(os.listdir(dest_folder))
                for item in sorted_items:
                    if not item.endswith('.raw'):
                        continue
                    if dest_folder[-1] != '/':
                       dest_folder = dest_folder+"/"
                    print(dest_folder + item)

                    path = dest_folder+item

                    #raw = rawpy.imread(path)
                    #image = raw.postprocess()
                    image = np.fromfile(dest_folder+item,dtype='float32')#, sep="")
                    print("shape im: ", image.shape)
                    X = int(path.split("_")[-2])
                    Y = int(path.split("_")[-1].split('.')[0])
                    if dim_invert:
                        xtemp = X
                        X = Y
                        Y = xtemp


                    image = np.reshape(image, (X,Y))

                    #if len(image.shape) == 2:
                    #    image = image.astype(np.float32)
                    #    image = np.expand_dims(img, axis=2)

                    #if dim_invert:
                    #    image = np.transpose(image, [1,0])

                    if color_invert:
                        mask = np.ones(image.shape)
                        #min_val = np.min(image)
                        #mask_up = mask + min_val
                        image = mask - image + 0.0000000001
                        #import cv2
                        #image = cv2.bitwise_and(image,image)#(cv2.bitwise_not(image))
                        #image = cv2.bitwise_and(image, mask_not)
                        #image[image <= 0] = 0.00000001
                    #image.astype('float32')

                    #import matplotlib.pyplot as plt
                    #plt.imshow(image, cmap="gray")
                    #plt.show()
                    #image.tofile(dest_folder+item+'.tiff')
                    all_images.append(image)
                    #if invert_image:
                    #    raw_image = invert(raw_image)
                    #f write_path:
                    #    raw_path = os.path.join(write_path.rsplit(".", 1)[0], 'raw_images', img_name)
                    #else:
                    #    raw_path = image_filename.rsplit('/', 1)[1].rsplit('.', 1)[0]
                    #image, fname_raw = blur_and_save(raw_image, raw_path + 'PERS' + str(persistence), blur_sigma=blur_sigma,
                    #                                 grey_scale=grey_scale)

            return all_images

        def transpose_first_index(self, x, with_hand_seg=False):
            if not with_hand_seg:
                x2 = (np.transpose(x[0], [2, 0, 1]), np.transpose(x[1], [2, 0, 1]), np.transpose(x[2], [2, 0, 1]))
            else:
                x2 = (np.transpose(x[0], [2, 0, 1]), x[1], np.transpose(x[2], [2, 0, 1]),
                      np.transpose(x[3], [2, 0, 1]))
            return x2


# Dataset class for the retina dataset
# each item of the dataset is a tuple with four items:
# - the first element is the input image to be segmented
# - the second element is the collection of morse smale complexes at various persistence and blur values
# - the third element is the segmentation ground truth image
# - the fourth element is a mask to know what parts of the input image should be used (for training and for scoring)
class MSCRetinaDataset(MSCLearningDataSet):#Dataset):
    def transpose_first_index(self, x, with_hand_seg=False):
        if not with_hand_seg:
            x2 = (np.transpose(x[0], [2, 0, 1]), np.transpose(x[1], [2, 0, 1]), np.transpose(x[2], [2, 0, 1]))
        else:
            x2 = (np.transpose(x[0], [2, 0, 1]), x[1], np.transpose(x[2], [2, 0, 1]),
                  np.transpose(x[3], [2, 0, 1]))
        return x2

    def __init__(self, retina_array=None, split='train', do_transform=False
                 , with_hand_seg=False, shuffle=True):
        super(MSCRetinaDataset, self).__init__()
        self.with_hand_seg = with_hand_seg
        if retina_array is not None:
            indexes_this_split = get_split(np.arange(len(retina_array), dtype=np.int), split, shuffle=shuffle)
            self.retina_array = [self.transpose_first_index(retina_array[i], self.with_hand_seg) for i in
                                 indexes_this_split]
        self.split = split
        self.do_transform = do_transform

    def __getitem__(self, index):
        sample = [x for x in self.retina_array[index]] #torch.FloatTensor(x)
        """if self.do_transform:
            v_gen = RandomVerticalFlipGenerator()
            h_gen = RandomHorizontalFlipGenerator()
            t = Compose([
                v_gen,
                RandomVerticalFlip(gen=v_gen),
                h_gen,
                RandomHorizontalFlip(gen=h_gen),
            ])
            sample = t(sample)"""
        return sample

    def __len__(self):
        return len(self.retina_array)

    # unzips, loads and calculates masks for images from the stare dataset
    def stare_read_images(self, tar_filename, dest_folder, do_mask=False):
        # tar = tarfile.open(tar_filename)
        # tar.extractall(dest_folder)
        # tar.close()
        all_images = []
        all_masks = []
        for item in sorted(os.listdir(dest_folder)):
            if dest_folder[-1] != '/':
               dest_folder = dest_folder+"/"
            if item.endswith('gz'):
                with gzip.open(dest_folder + item, 'rb') as f_in:
                    with open(dest_folder + item[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(dest_folder + item)
            img = imageio.imread(os.path.join(dest_folder, item))#[:-3])
            """
            if len(img.shape) == 2:
                img = img.astype(np.float32)
                img = np.expand_dims(img, axis=2)
            all_images.append(img)
            if do_mask:
                mask = (1 - remove_small_regions(np.prod((img < 50 / 255.) * 1.0, axis=2) > 0.5, 1000)) * 1.0
                mask = np.expand_dims(mask, axis=2)
                all_masks.append(mask.astype(np.float32))
            """
            if len(img.shape) == 3:
                img = np.pad(img, ((1, 2), (2, 2), (0, 0)), mode='constant')
            else:
                img = np.pad(img, ((1, 2), (2, 2)), mode='constant')
            #img = resize_img(img)
            img = img / 255.
            img = img.astype(np.float32)
            if len(img.shape) == 2:
                img = img.astype(np.float32)
                img = np.expand_dims(img, axis=2)
            all_images.append(img)
            if do_mask:
                mask = (1 - remove_small_regions(np.prod((img < 50 / 255.) * 1.0, axis=2) > 0.5, 1000)) * 1.0
                mask = np.expand_dims(mask, axis=2)
                all_masks.append(mask.astype(np.float32))

        if do_mask:
            return all_images, all_masks
        else:
            return all_images

    # unzips and loads masks for images from the stare dataset
    def drive_read_images(self, filetype, dest_folder):
        # zip_ref = zipfile.ZipFile('DRIVE.zip', 'r')
        # zip_ref.extractall('datasets/drive')
        # zip_ref.close()
        all_images = []
        sorted_items = sorted(os.listdir(dest_folder))
        for item in sorted_items:
            if dest_folder[-1] != '/':
               dest_folder = dest_folder+"/"
            if item.endswith(filetype):
                img = imageio.imread(dest_folder + item)
                if len(img.shape) == 3:
                    img = np.pad(img, ((12, 12), (69, 70), (0, 0)), mode='constant')
                else:
                    img = np.pad(img, ((12, 12), (69, 70)), mode='constant')
                #img = resize_img(img)
                img = img / 255.
                img = img.astype(np.float32)
                if len(img.shape) == 2:
                    img = img.astype(np.float32)
                    img = np.expand_dims(img, axis=2)
                all_images.append(img)
                #print("drive read im shape ", img.shape)
        return all_images

    # load all images and put them on a list of list of arrays.
    # on the inner lists, first element is an input image, second element is a segmentation groundtruth
    # and third element is a mask to show where the input image is valid, in contrast to where it was padded


    def get_retina_array(self, partial=False, use_local_setup=True, msc=True
                         , stare_only=False, drive_training_only=False, drive_test_only=False
                         ,persistence_values=[], env='multivax', number_images=None):

        if use_local_setup:
            self.LocalSetup = LocalSetup(env)

            drive_training_path = self.LocalSetup.drive_training_path
            drive_segmentation_path = self.LocalSetup.drive_training_segmentation_path
            # training_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/neuron_msc"
            drive_test_path = self.LocalSetup.drive_test_path
            drive_training_mask_path = self.LocalSetup.drive_training_mask_path
            # "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/" #
            drive_test_segmentation_path = self.LocalSetup.drive_test_segmentation_path
            drive_test_mask_path = self.LocalSetup.drive_test_mask_path
            stare_image_path = self.LocalSetup.stare_training_data_path
            stare_segmentation_path = self.LocalSetup.stare_segmentations
            drive_training_msc_segmentation_path = self.LocalSetup.drive_training_msc_segmentation_path
            stare_msc_segmentation_path = self.LocalSetup.stare_msc_segmentation_path
        else:
            stare_image_path = 'datasets/stare/images/'
            stare_segmentation_path = 'datasets/stare/segmentations/'
            drive_segmentation_path = 'datasets/drive/DRIVE/training/1st_manual/'
            drive_test_path = 'datasets/drive/DRIVE/test/images/'
            drive_training_path = 'datasets/drive/DRIVE/training/images/'
            drive_training_mask_path = 'datasets/drive/DRIVE/training/mask/'
            drive_test_segmentation_path = 'datasets/drive/DRIVE/test/1st_manual/'
            drive_test_mask_path = 'datasets/drive/DRIVE/test/mask/'
            drive_training_msc_segmentation_path = 'datasets/drive/DRIVE/training/msc_seg'
            stare_msc_segmentation_path = 'datasets/stare/training/msc_seg'

        self.number_persistence_vals = len(persistence_values)
        stare_images, stare_mask = self.stare_read_images("ppm", stare_image_path, do_mask=True)
        stare_segmentation = self.stare_read_images("ppm", stare_segmentation_path)
        drive_training_images = self.drive_read_images('tif', drive_training_path)
        drive_training_mask = self.drive_read_images('gif', drive_training_mask_path)
        drive_test_images = self.drive_read_images('tif', drive_test_path)

        drive_training_segmentations = self.drive_read_images('gif', drive_segmentation_path)

        # hand draw ground truth
        drive_test_segmentation = self.drive_read_images('gif', drive_test_segmentation_path)
        drive_test_mask = self.drive_read_images('gif', drive_test_mask_path)

        # collect msc if msc already computed.
        if msc:
            if False:
                for i in range(len(persistence_values)):
                    stare_mask += stare_mask
                    stare_images += stare_images
                    drive_training_images += drive_training_images
                    drive_training_mask += drive_training_mask
                    drive_training_segmentations += drive_training_segmentations

            # collect pre-computed msc from directories
            drive_training_msc_segmentations = [os.path.join(drive_training_msc_segmentation_path, o)
                                                       for o in os.listdir(drive_training_msc_segmentation_path)
                                                       if os.path.isdir(os.path.join(drive_training_msc_segmentation_path, o))]

            drive_training_msc = []
            for msc_seg in sorted(drive_training_msc_segmentations):
                msc_group = self.drive_read_images('tif', msc_seg)
                drive_training_msc += msc_group

            # stare
            stare_msc_segmentations = [os.path.join(stare_msc_segmentation_path, o) for o in
                                                   os.listdir(stare_msc_segmentation_path) if os.path.isdir(os.path.join(stare_msc_segmentation_path, o))]

            stare_msc = []
            for msc_seg in sorted(stare_msc_segmentations):
                msc_group = self.drive_read_images('tif', msc_seg)
                stare_msc += msc_group

            #stare_training_msc_segmentation = stare_training_segmentation_msc_pers_7 + stare_training_segmentation_msc_pers_10 + stare_training_segmentation_msc_pers_12 + stare_training_segmentation_msc_pers_15 + stare_training_segmentation_msc_pers_20 + stare_training_segmentation_msc_pers_23

            #all_stare_msc_seg = stare_training_segmentation_msc_pers_7 + stare_training_segmentation_msc_pers_10 + stare_training_segmentation_msc_pers_12 + stare_training_segmentation_msc_pers_15 + stare_training_segmentation_msc_pers_20 + stare_training_segmentation_msc_pers_23 + stare_training_segmentation_msc_pers_30  # + stare_training_segmentation_msc_pers_25
            # stare_training_segmentation_msc_pers_10 + stare_training_segmentation_msc_pers_12
            if drive_training_only:
                total_msc_segmentation = drive_training_msc
            elif stare_only:
                total_msc_segmentation = stare_msc
            else:
                total_msc_segmentation = stare_msc + drive_training_msc
        else:
            #dummy space filler
            if drive_training_only:
                total_msc_segmentation = drive_training_segmentations
            elif stare_only:
                total_msc_segmentation = stare_segmentation
            elif drive_test_only:
                total_msc_segmentation = []
            else:
                total_msc_segmentation = stare_segmentation + drive_training_segmentations

        if drive_training_only:
            if number_images is None:
                self.retina_array = list(zip(drive_training_images,
                                             total_msc_segmentation,
                                             drive_training_mask,
                                             drive_training_segmentations))
            else:
                self.retina_array = list(zip(drive_training_images[:number_images],
                                             total_msc_segmentation[:number_images],
                                             drive_training_mask[:number_images],
                                             drive_training_segmentations[:number_images]))
            return self.retina_array
        if drive_test_only:
            if number_images is None:
                self.retina_array = list(zip(drive_test_images,
                                             drive_test_segmentation,
                                             drive_test_mask,
                                             drive_test_segmentation))
            else:
                self.retina_array = list(zip(drive_test_images[:number_images],
                                             drive_test_segmentation[:number_images],
                                             drive_test_mask[:number_images],
                                             drive_test_segmentation[:number_images]))
            return self.retina_array
        if stare_only:
            if number_images is None:
                self.retina_array = list(zip(stare_images,
                                             total_msc_segmentation,
                                             stare_mask,
                                             stare_segmentation))
            else:
                self.retina_array = list(zip(stare_images[:number_images],
                                             total_msc_segmentation[:number_images],
                                             stare_mask[:number_images],
                                             stare_segmentation[:number_images]))
            return self.retina_array
        if partial:
            if number_images is None:
                self.retina_array = list(zip(stare_images + drive_training_images + drive_test_images,
                                total_msc_segmentation + drive_test_segmentation,
                                stare_mask + drive_training_mask + drive_test_mask))
            else:
                self.retina_array = list(zip(stare_images[:number_images] + drive_training_images[:number_images] + drive_test_images[:number_images],
                                             total_msc_segmentation[:number_images] + drive_test_segmentation[:number_images],
                                             stare_mask[:number_images] + drive_training_mask[:number_images] + drive_test_mask[:number_images]))
        else:
            if number_images is None:
                self.retina_array = list(zip(stare_images + drive_training_images + drive_test_images,
                                total_msc_segmentation + drive_test_segmentation,
                                stare_mask + drive_training_mask + drive_test_mask,
                                stare_segmentation + drive_training_segmentations + drive_test_segmentation))
            else:
                self.retina_array = list(zip(stare_images[:number_images] + drive_training_images[:number_images] + drive_test_images[:number_images],
                                             total_msc_segmentation[:number_images] + drive_test_segmentation[:number_images],
                                             stare_mask[:number_images] + drive_training_mask[:number_images] + drive_test_mask[:number_images],
                                             stare_segmentation[:number_images] + drive_training_segmentations[:number_images] + drive_test_segmentation[:number_images]))
        return self.retina_array

class Neuron2dDataSet(MSCLearningDataSet):

    def __init__(self, data_array=None, dataset='neuron2', split='train', do_transform=False
                 , with_hand_seg=False, shuffle=True, dim_invert = False):
        super().__init__()

        self.dataset = dataset
        self.dim_invert = dim_invert
        #self.with_hand_seg = with_hand_seg
        if data_array is not None:
            self.data_array = data_array
        else:
            self.data_array = self._get_neuron_data()
        #if data_array is not None:
        #indexes_this_split = get_split(np.arange(len(data_array), dtype=np.int), split, shuffle=shuffle)
        #self.data_array = [self.transpose_first_index(data_array[i], self.with_hand_seg) for i in
        #                     indexes_this_split]
        #self.split = split
        #self.do_transform = do_transform


    def get_neuron_data(self):
        return self.data_array

    def _get_neuron_data(self, number_images=None):

        data_folder = "/home/sam/Documents/PhD/Research/GeoMSCxML/datasets/" + str(self.dataset)

        images = super().read_images(filetype='raw', dest_folder=data_folder,color_invert=False, dim_invert=self.dim_invert)
        msc_list = images
        mask= [None]
        training_msc_list = [None]

        if number_images is None:
            self.data_array = list(zip(images,
                                       msc_list,
                                       mask,
                                       training_msc_list))
        else:
            self.data_array = list(zip(images[:number_images],
                                       msc_list,
                                       mask,
                                       training_msc_list))
        return self.data_array