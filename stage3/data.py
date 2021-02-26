import numpy as np
import cv2

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools

train_boarder = 112
rotate_center = (55,55)

def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    #pixels = (img - mean)*1e+3
    return pixels


def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    isface = int(line_parts[5])
    # now this could be empty if isface==0
    landmarks = list(map(float, line_parts[6: len(line_parts)]))
    return img_name, rect, isface, landmarks


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        image, landmarks, isface = sample['image'], sample['landmarks'], sample['isface']
        image_resize = np.asarray(
                            image.resize((train_boarder, train_boarder), Image.BILINEAR),
                            dtype=np.float32)       # Image.ANTIALIAS)
        image = channel_norm(image_resize)
        return {'image': image,
                'landmarks': landmarks,
                'isface': isface
                }

class Rotation_allangle(object):
    """
       Randomly rotate the image and the corresponding landmarks
       Need to do Normalize, which does image resizing
    """
    def __call__(self,sample):
        image, landmarks, isface = sample['image'], sample['landmarks'], sample['isface']

        alpha = np.random.rand()*365.0
        scale1 = 1.0/(np.abs(np.sin(np.deg2rad(alpha+45.0))))
        scale2 = 1.0/(np.abs(np.sin(np.deg2rad(alpha-45.0))))
        scale = np.min([scale1,scale2])/1.4142

        # the scaling factor ensure the entire image is still with 112x112
        M = cv2.getRotationMatrix2D(rotate_center,
                  alpha,scale)

        # rotate the image
        image_rotate = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))

        # rotate the landmarks
        # the landmarks have been normalized
        landmarks_rotate = np.concatenate((np.reshape(landmarks,(21,2))*train_boarder,
                                           np.ones((21,1))),axis=1) @ np.transpose(M) 
        landmarks_rotate = np.reshape(landmarks_rotate/train_boarder,(42)).astype(np.float32)
        return {'image': image_rotate,
                'landmarks': landmarks_rotate,
                'isface': isface
                }

class Rotation_flip(object):
    """
       Randomly rotate the image and the corresponding landmarks
       Need to do Normalize, which does image resizing
    """
    def __call__(self,sample):
        image, landmarks, isface = sample['image'], sample['landmarks'], sample['isface']

        alpha = np.random.choice([0,90,180,270])
        scale1 = 1.0/(np.abs(np.sin(np.deg2rad(alpha+45.0))))
        scale2 = 1.0/(np.abs(np.sin(np.deg2rad(alpha-45.0))))
        scale = np.min([scale1,scale2])/1.4142

        # the scaling factor ensure the entire image is still with 112x112
        M = cv2.getRotationMatrix2D(rotate_center,
                  alpha,scale)

        # rotate the image
        image_rotate = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))

        # rotate the landmarks
        # the landmarks have been normalized
        landmarks_rotate = np.concatenate((np.reshape(landmarks,(21,2))*train_boarder,
                                           np.ones((21,1))),axis=1) @ np.transpose(M)                 
        landmarks_rotate = np.reshape(landmarks_rotate/train_boarder,(42)).astype(np.float32)
        #landmarks_rotate = np.concatenate((np.reshape(landmarks,(21,2)),
        #                                   np.ones((21,1))),axis=1) @ np.transpose(M)                 
        #landmarks_rotate = np.reshape(landmarks_rotate,(42)).astype(np.float32)
        return {'image': image_rotate,
                'landmarks': landmarks_rotate,
                'isface': isface
                }

class Rotation_none(object):
    """
       Randomly rotate the image and the corresponding landmarks
       Need to do Normalize, which does image resizing
    """
    def __call__(self,sample):
        image, landmarks, isface = sample['image'], sample['landmarks'], sample['isface']
        return {'image': image,
                'landmarks': landmarks,
                'isface': isface
                }


def unnormalize_forplotting(img):
    img_out = img.numpy()[0,:,:]
    img_out = img_out - np.min(img_out)
    fac = 255.0 / np.max(img_out) 
    img_out = img_out * fac
    img_out = img_out.astype('uint8')
    return img_out

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        image, landmarks, isface = sample['image'], sample['landmarks'], sample['isface']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)

        # it needs this for batching
        #print(landmarks.shape)
        if isface == False:
            landmarks = np.zeros(42,dtype=np.float32)

        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks),
                'isface': isface}


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, isface, landmarks = parse_line(self.lines[idx])
        # image
        img = Image.open(img_name).convert('L')     
        img_crop = img.crop(tuple(rect))            
        landmarks = np.array(landmarks).astype(np.float32)
		
		
	# you should let your landmarks fit to the train_boarder(112)
	# please complete your code under this blank
	# your code:
		
        W = rect[2] - rect[0]
        H = rect[3] - rect[1]
        for idx in range(len(landmarks)):
            if idx%2 == 0:
                #landmarks[idx] = landmarks[idx] / W * train_boarder
                landmarks[idx] = landmarks[idx] / W
            else:
                #landmarks[idx] = landmarks[idx] / H * train_boarder
                landmarks[idx] = landmarks[idx] / H
		

        sample = {'image': img_crop, 'landmarks': landmarks, 'isface': isface}
        sample = self.transform(sample)
        return sample


def load_data(phase,rotation_class):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),                # do channel normalization
            rotation_class(),                 # do rotation
            ToTensor()]                 # convert to torch type: NxCxHxW
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, transform=tsfm)
    return data_set


def get_train_test_set(rotation_class):
    train_set = load_data('train',rotation_class)
    valid_set = load_data('test',rotation_class)
    return train_set, valid_set


if __name__ == '__main__':
    rotation_class = Rotation_none
    train_set = load_data('train',rotation_class)
    for sample in train_set:
        img = sample['image']
        landmarks = sample['landmarks']
        isface = sample['isface']
 
        print("checking:",img,landmarks,isface)

	    #画出人脸crop以及对应的landmarks
        img_out = unnormalize_forplotting(img)
        for idx in range(0,len(landmarks),2):
            cv2.circle(img_out, (landmarks[idx]*train_boarder,landmarks[idx+1]*train_boarder), 2, (0,0,255))
            #cv2.circle(img_out, (landmarks[idx],landmarks[idx+1]), 2, (0,0,255))
        cv2.imshow('data',img_out)		

        key = cv2.waitKey()
        if key == 27:
            exit(0)
        cv2.destroyAllWindows()

