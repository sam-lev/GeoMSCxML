import os
##################################
#
#           FILE STRUCTURE
# Home
#   \_GeoMSCxML
#        \_datasets
#           \_optics \_stare
#             
##################################
class LocalSetup:
        def __init__(self):
            # Paths for Multivax
            multivax = False
            sci = not multivax
            self.paths_for_multivax = """ training_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/drive/DRIVE/training/images"
                        #training_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/neuron_msc"
                        testing_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/drive/DRIVE/test/images"
                        train_write_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/drive/DRIVE/training/" # "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/" #
                        test_write_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/drive/DRIVE/test/"
                        stare_training_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/stare/images"
                        stare_train_write_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/HW3/datasets/stare/"
                        """
            if multivax:
                self.project_base_path = "/home/sam/Documents/PhD/Research/GeoMSCxML/"
                self.stare_base_path = os.path.join(self.project_base_path, "datasets", "optics", "stare")
                self.drive_test_base_path = os.path.join(self.project_base_path, "datasets", "optics", "drive", "DRIVE", "test")
                self.drive_training_base_path = os.path.join(self.project_base_path, "datasets", "optics", "drive", "DRIVE","training")

                self.drive_training_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","training","images")
                self.drive_training_mask_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","training","mask")
                self.drive_test_segmentation_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","test","1st_manual")
                self.drive_test_mask_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","test","mask")
                self.drive_training_segmentation_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","training","1st_manual")
                #training_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/neuron_msc"
                self.drive_test_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","test","images")
                self.drive_write_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","training")
                self.stare_write_path = os.path.join(self.project_base_path, "datasets", "optics",   "stare" )
                # "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/" #
                self.test_write_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","test")

                self.stare_training_data_path = os.path.join(self.project_base_path,"datasets","optics","stare","images")
                self.stare_segmentations = os.path.join(self.project_base_path,"datasets","optics","stare", "segmentations")

                #msc paths
                self.drive_training_msc_segmentation_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","training","msc_seg")
                self.stare_msc_segmentation_path = os.path.join(self.project_base_path, "datasets", "optics","stare", "msc_seg")


            ## Paths for SCI  ##
            ####################
            if sci:

                self.project_base_path = "/home/sci/samlev/GeoMSCxML/"
                self.stare_base_path = os.path.join(self.project_base_path, "datasets", "optics", "stare")
                self.drive_test_base_path = os.path.join(self.project_base_path, "datasets", "optics", "drive", "DRIVE", "test")
                self.drive_training_base_path = os.path.join(self.project_base_path, "datasets", "optics", "drive", "DRIVE","training")

                self.drive_training_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","training","images")
                self.drive_training_mask_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","training","mask")
                self.drive_test_segmentation_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","test","1st_manual")
                self.drive_test_mask_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","test","mask")
                self.drive_training_segmentation_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","training","1st_manual")
                #training_data_path = "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/neuron_msc"
                self.drive_test_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","test","images")
                self.drive_write_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","training")
                self.stare_write_path = os.path.join(self.project_base_path, "datasets", "optics",   "stare" )
                # "/Users/multivax/Documents/PhD/4spring19/DeepLearning/DeepLearning/final_project/results/" #
                self.test_write_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","test")

                self.stare_training_data_path = os.path.join(self.project_base_path,"datasets","optics","stare","images")
                self.stare_segmentations = os.path.join(self.project_base_path,"datasets","optics","stare", "segmentations")

                #msc paths
                self.drive_training_msc_segmentation_path = os.path.join(self.project_base_path,"datasets","optics","drive","DRIVE","training","msc_seg")
                self.stare_msc_segmentation_path = os.path.join(self.project_base_path, "datasets", "optics","stare", "msc_seg")
