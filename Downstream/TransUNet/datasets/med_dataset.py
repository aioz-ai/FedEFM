import os
import random
#import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import logging 
import cv2

from skimage.transform import resize

def random_rot_flip(image, label, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self,seed, output_size):
        self.output_size = output_size
        self.seed  = seed
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        k = random.random()
        if k > 0.5:
            image, label = random_rot_flip(image, label, self.seed)
        else:
            image, label = random_rotate(image, label, self.seed)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

    

class MedDataset(Dataset):
    def __init__(self, images_dir, mask_dir, scale = (512,512), transform=None):
        self.images_dir = images_dir  #med
        self.scale = scale
        self.masks_dir = mask_dir
        self.ids = os.listdir(images_dir) # list contains all images
        if len(self.ids) == 0:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.cache = {}
        self.transform = transform

    def __len__(self):
        return len(self.ids)



    def __getitem__(self, idx):
        image_name = self.ids[idx]
        mask_name  = image_name.split(".")[0]+".npy"


        image_file = os.path.join(self.images_dir, image_name)
        mask_file = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        mask = np.load(mask_file)

        image = resize(image, 
                     (self.scale[0], self.scale[0]), 
                     order=0, 
                     preserve_range=True, 
                     anti_aliasing=False).astype('uint8')
        image = np.asarray(image)
        image = ((image - image.min()) * (1/(0.01 + image.max() - image.min()) * 255)).astype('uint8')
        
        mask = resize(mask , 
                         (self.scale[1], self.scale[1]), 
                         order=0, 
                         preserve_range=True, 
                         anti_aliasing=False).astype('uint8')



        sample = {'image': image, 'label': mask}
        if self.transform:
            sample = self.transform(sample)
        sample['name'] = image_name
        return sample
    

class SimulationDataset(Dataset):
    def __init__(self, images_dir, mask_dir, scale = (512,512), transform=None):
        self.images_dir = images_dir  #med
        self.scale = scale
        self.masks_dir = mask_dir
        self.ids = os.listdir(images_dir) # list contains all images
        if len(self.ids) == 0:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.cache = {}
        self.transform = transform

    def __len__(self):
        return len(self.ids)



    def __getitem__(self, idx):
        image_name = self.ids[idx]
        #mask_name  = image_name.split(".")[0]+".npy"
        mask_name = image_name.split(".")[0] + "_mask.png"

        image_file = os.path.join(self.images_dir, image_name)
        mask_file = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        #mask = np.load(mask_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = (mask>0).astype(np.uint8)
        image = resize(image, 
                     (self.scale[0], self.scale[0]), 
                     order=0, 
                     preserve_range=True, 
                     anti_aliasing=False).astype('uint8')
        image = np.asarray(image)
        image = ((image - image.min()) * (1/(0.01 + image.max() - image.min()) * 255)).astype('uint8')
        
        mask = resize(mask , 
                         (self.scale[1], self.scale[1]), 
                         order=0, 
                         preserve_range=True, 
                         anti_aliasing=False).astype('uint8')



        sample = {'image': image, 'label': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample
