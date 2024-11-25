import numpy as np
import nibabel as nib
import glob
import os
from tensorflow.keras.utils import to_categorical
import splitfolders
from sklearn.preprocessing import MinMaxScaler

#####################
# Script parameters #
#####################
TRAIN_PATH = 'C:/Users/ygrae/Desktop/BRATS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData' # e.g. 'C:/Users/.../BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
TRAIN_SPLIT = 0.8

# Create list containing paths of all images and masks from training downloaded training set
t2_list = sorted(glob.glob(os.path.join(TRAIN_PATH, '**', '*_t2.nii'), recursive=True))
t1ce_list = sorted(glob.glob(os.path.join(TRAIN_PATH, '**', '*_t1ce.nii'), recursive=True))
flair_list = sorted(glob.glob(os.path.join(TRAIN_PATH, '**', '*_flair.nii'), recursive=True))
mask_list = sorted(glob.glob(os.path.join(TRAIN_PATH, '**', '*_seg.nii'), recursive=True))

# Ensure all modality lists and the mask list have the same length
if len(t2_list) == len(t1ce_list) == len(flair_list) == len(mask_list):
    pass  # All lists are of the same length
else:
    raise ValueError("Mismatch in the number of files between T2, T1CE, Flair, and Segmentation masks.")

# Create new folders to save numpy arrays created below
base_path_one_up = os.path.dirname(TRAIN_PATH)

path_images = os.path.join(base_path_one_up, 'input_data_3channels/images/')
os.makedirs(path_images, exist_ok=True)

path_masks = os.path.join(base_path_one_up, 'input_data_3channels/masks/')
os.makedirs(path_masks, exist_ok=True)

# Pre-process images and masks
scaler = MinMaxScaler()

for img in range(len(t2_list)):   
    print("Now preparing image and masks number: ", img)
    
    # load images as np arrays and scale them with MinMaxScaler (repeat for the 3 different channels)
    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
   
    temp_image_flair=nib.load(flair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

    # load mask as np array and reassign values of 4 to 3
    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  
    
    # stack the created numpy arrays
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    
    # Crop to a size divisible by 64 (i.e. 128) so we can later extract 64x64x64 patches
    # We don't lose info bc we only crop away the dark part of the images (brain MRI is in center)
    #cropping x, y, and z
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    # only keep relevant image data (as to not waste computations) and save them in new folder
    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0]/counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0. Otherwise don't bother saving numpy arrays
        temp_mask= to_categorical(temp_mask, num_classes=4)
        np.save(path_images+'image_'+str(img)+'.npy', temp_combined_images)
        np.save(path_masks+'mask_'+str(img)+'.npy', temp_mask)
        
    else:
        pass 

# split data into training and validation set
input_path = os.path.join(base_path_one_up, 'input_data_3channels/')
output_path = os.path.join(base_path_one_up, 'input_data128/')

splitfolders.ratio(input=input_path, output=output_path, 
                   seed=42,ratio=(TRAIN_SPLIT,1-TRAIN_SPLIT),
                   group_prefix=None)


