# specifications

#IMAGES_FOLDER_PATH = "/MLDatasetsStorage/Perceptron_Master_Dataset/Perceptron_Coarse_Training/images/val/BDD100k"
IMAGES_FOLDER_PATH = "C:/Users/s26443/Desktop/BDD100k_finetune/images/test"
LABELS_FOLDER_PATH = "C:/Users/s26443/Desktop/BDD100k_finetune/labels/test"
#SAVING_DIRECTORY = "/MLDatasetsStorage/Perceptron_analysis/"
SAVING_DIRECTORY = "C:/Users/s26443/git-repos/pytorch_training/save/"

#WEIGHTS_PATH = "/WeightModels/PerceptronWeights/SegNet_public_finetune/SegNet"
WEIGHTS_PATH = "C:/Users/s26443/git-repos/pytorch_training/save/SegNet_public_finetune/SegNet"
NBR_CLASSES = 2
MODEL_NAME = "segnet"
NBR_IMAGES_TO_COMPARE = 10 #number of original images on which to apply noise
NOISE_BATCH = 20 # nbr of noise matrices to apply to each image
INPUT_SHAPE = (480, 360, 3)


# AVAILABLE NOISE TYPES
NOISE_TYPE = "pepper"