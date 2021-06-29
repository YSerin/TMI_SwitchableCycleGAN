import os
import random
from glob import glob

import numpy as np
import pydicom
import scipy.io as sio
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from scipy import io
import ipdb

class Dataset_fold_best(data.Dataset):

    def __init__(self, opt, phase, test_index = None, size='whole'):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.phase = phase
        self.size = size

        self.path_H = []
        self.path_S = []
        dir_id_list = self.opt.data_dir + '{}_total'.format(self.opt.data_type)
        dir_id_list_train = self.opt.data_dir + '{}_train'.format(self.opt.data_type)
        dir_id_list_test = self.opt.data_dir + '{}_test'.format(self.opt.data_type)
        self.id_list = sorted(os.listdir(dir_id_list))
        self.id_list_train = sorted(os.listdir(dir_id_list_train))
        self.dir = sorted(glob(dir_id_list_train+'/*'))
        if self.phase == 'train':
            if self.size == 'whole':
                self.dir.pop(test_index + 5)
                self.dir.pop(test_index)
            elif self.size == 'half':
                if test_index == 0:
                    self.dir = [self.dir[3], self.dir[4], self.dir[8], self.dir[9]]
                elif test_index == 1:
                    self.dir = [self.dir[0], self.dir[4], self.dir[5], self.dir[9]]
                elif test_index == 2:
                    self.dir = [self.dir[0], self.dir[1], self.dir[5], self.dir[6]]
                elif test_index == 3:
                    self.dir = [self.dir[1], self.dir[2], self.dir[6], self.dir[7]]
                elif test_index == 4:
                    self.dir = [self.dir[2], self.dir[3], self.dir[7], self.dir[8]]
                else:
                    raise ValueError
            elif self.size == 'quarter':
                if test_index+6 == 10:
                    self.dir = [self.dir[0], self.dir[5]]
                else:
                    self.dir = [self.dir[test_index + 1], self.dir[test_index+6]]
        elif self.phase == 'validation':
            self.dir = [self.dir[test_index], self.dir[test_index+5]]
        elif self.phase == 'test':
            self.dir = sorted(glob(dir_id_list_test+'/*'))

        # Training Set
        if self.phase == 'train':
            self.img_crop_size = self.opt.img_crop_size

        for p in self.dir:
            tmp_path_H = glob(p + '/H/*.dcm')
            tmp_path_S = glob(p + '/S/*.dcm')

            self.path_H.extend(tmp_path_H)
            self.path_S.extend(tmp_path_S)

        self.path_H = sorted(self.path_H)
        self.path_S = sorted(self.path_S)

        dir_stat = self.opt.data_dir + 'statistic_{}'.format(self.opt.data_type)
        self.mean_H = np.load(dir_stat + '/mean_H.npz')['meanlist']
        self.mean_S = np.load(dir_stat + '/mean_S.npz')['meanlist']
        self.std_H = np.load(dir_stat + '/std_H.npz')['stdlist']
        self.std_S = np.load(dir_stat + '/std_S.npz')['stdlist']

    def __getitem__(self, index):

        path_H = self.path_H[index]
        if self.phase == 'train':
            path_S = self.path_S[len(self.path_H)-index-1]
        else:
            path_S = self.path_S[index]

        img_H = _read_dicom(path_H)
        img_S = _read_dicom(path_S)


        if (self.phase == 'train' or self.phase == 'validation') and self.opt.data_type == 'Head':
            id_H = path_H[46:54]
            id_S = path_S[46:54]
        elif (self.phase == 'train' or self.phase == 'validation') and self.opt.data_type == 'Facial_bone':
            id_H = path_H[53:61]
            id_S = path_S[53:61]
        elif self.phase == 'test' and self.opt.data_type == 'Head':
            id_H = path_H[45:53]
            id_S = path_S[45:53]
        elif self.phase == 'test' and self.opt.data_type == 'Facial_bone':
            id_H = path_H[52:60]
            id_S = path_S[52:60]
        else:
            raise ValueError


        # normalize for each volume
        img_H -= self.mean_H[self.id_list.index(str(id_H))]
        img_S -= self.mean_S[self.id_list.index(str(id_S))]
        img_H /= self.std_H[self.id_list.index(str(id_H))]
        img_S /= self.std_S[self.id_list.index(str(id_S))]

        if self.phase == 'train':
            img_H = Image.fromarray(img_H)
            img_S = Image.fromarray(img_S)

            # Random Horizontal flip for Paired Images
            if random.random() > 0.5:
                img_S = transforms.functional.vflip(img_S)
            if random.random() > 0.5:
                img_H = transforms.functional.vflip(img_H)
            if random.random() > 0.5:
                img_S = transforms.functional.hflip(img_S)
            if random.random() > 0.5:
                img_H = transforms.functional.hflip(img_H)

            i, j, h, w = transforms.RandomCrop.get_params(img_S, output_size=(self.img_crop_size, self.img_crop_size))
            img_S = np.array(transforms.functional.crop(img_S, i, j, h, w))
            img_H = np.array(transforms.functional.crop(img_H, i, j, h, w))

            # img_S = _random_crop(img_S, output_size = self.img_crop_size)
            # img_H = _random_crop(img_H, output_size = self.img_crop_size)

        img_S = img_S[np.newaxis, :, :]
        img_H = img_H[np.newaxis, :, :]

        img_S = torch.tensor(img_S, dtype=torch.float32)
        img_H = torch.tensor(img_H, dtype=torch.float32)

        return {'real_S': img_S, 'real_H':img_H, 'id':[id_H,id_S], 'path_H':path_H, 'path_S':path_S}

    def __len__(self):
        return int(len(self.path_H))

# Generate H(Hr68),S(Hr40) and M(Hr49) kernel
class Dataset_3fold_best_M(data.Dataset):

    def __init__(self, opt, phase, test_index = None, size=None):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.phase = phase
        if test_index:
            test_index = test_index*2
        if phase == 'train':
            self.img_crop_size = self.opt.img_crop_size
            self.crop = transforms.RandomCrop(self.img_crop_size)

        self.path_H = []
        self.path_S = []
        self.path_M = []
        dir_id_list = self.opt.data_dir + '{}_total'.format(self.opt.data_type)
        dir_id_list_train = self.opt.data_dir + '{}_train'.format(self.opt.data_type)
        dir_id_list_test = self.opt.data_dir + '{}_test'.format(self.opt.data_type)
        self.id_list = sorted(os.listdir(dir_id_list))
        self.id_list_train = sorted(os.listdir(dir_id_list_train))
        self.dir = sorted(glob(dir_id_list_train+'/*'))
        if self.phase == 'train':
            if size == 'whole':     #5cases
                self.dir.pop(test_index+1)
                self.dir.pop(test_index)
            elif size == '3cases':
                if test_index == 4:
                    self.dir.pop((test_index + 2) % 7)
                    self.dir.pop((test_index + 1) % 7)
                    self.dir.pop((test_index) % 7)
                    self.dir.pop((test_index + 3) % 7)
                else:
                    self.dir.pop((test_index + 3) % 7)
                    self.dir.pop((test_index + 2) % 7)
                    self.dir.pop((test_index + 1) % 7)
                    self.dir.pop((test_index) % 7)
            elif size == '1case':
                self.dir = [self.dir[test_index+2]]
            else:
                raise ValueError
        elif self.phase == 'validation':
            self.dir = [self.dir[test_index], self.dir[test_index+1]]
        elif self.phase == 'test':
            self.dir = sorted(glob(dir_id_list_test+'/*'))

        # Training Set
        if self.phase == 'train':
            self.img_crop_size = self.opt.img_crop_size

        for p in self.dir:
            tmp_path_H = glob(p + '/Hr68/*.dcm')
            tmp_path_S = glob(p + '/Hr40/*.dcm')
            tmp_path_M = glob(p + '/Hr49/*.dcm')

            self.path_H.extend(tmp_path_H)
            self.path_M.extend(tmp_path_M)
            self.path_S.extend(tmp_path_S)

        self.path_H = sorted(self.path_H)
        self.path_M = sorted(self.path_M)
        self.path_S = sorted(self.path_S)

        dir_stat = self.opt.data_dir + 'statistic_{}'.format(self.opt.data_type)
        self.mean_H = np.load(dir_stat + '/mean_Hr68.npz')['meanlist']
        self.mean_S = np.load(dir_stat + '/mean_Hr40.npz')['meanlist']
        self.std_H = np.load(dir_stat + '/std_Hr68.npz')['stdlist']
        self.std_S = np.load(dir_stat + '/std_Hr40.npz')['stdlist']
        self.mean_M = np.load(dir_stat + '/mean_Hr49.npz')['meanlist']
        self.std_M = np.load(dir_stat + '/std_Hr49.npz')['stdlist']

        self.train_order_M = np.arange(0,len(self.path_M))
        self.train_order_S = np.arange(0,len(self.path_S))
        random.shuffle(self.train_order_M)
        random.shuffle(self.train_order_S)

    def __getitem__(self, index):

        path_H = self.path_H[index]
        if self.phase == 'train':
            path_S = self.path_S[self.train_order_S[index]]
            path_M = self.path_M[self.train_order_M[index]]
        else:
            path_S = self.path_S[index]
            path_M = self.path_M[index]

        img_H = _read_dicom(path_H)
        img_M = _read_dicom(path_M)
        img_S = _read_dicom(path_S)


        if (self.phase == 'train' or self.phase == 'validation') and self.opt.data_type == 'Facial_bone':
            id_H = path_H.split('/Facial_bone_train/')[1][:8]
            id_M = path_M.split('/Facial_bone_train/')[1][:8]
            id_S = path_S.split('/Facial_bone_train/')[1][:8]
        elif self.phase == 'test' and self.opt.data_type == 'Facial_bone':
            id_H = path_H.split('/Facial_bone_test/')[1][:8]
            id_M = path_M.split('/Facial_bone_test/')[1][:8]
            id_S = path_S.split('/Facial_bone_test/')[1][:8]
        else:
            raise ValueError

        # normalize for each volume
        img_H -= self.mean_H[self.id_list.index(str(id_H))]
        img_M -= self.mean_M[self.id_list.index(str(id_M))]
        img_S -= self.mean_S[self.id_list.index(str(id_S))]
        img_H /= self.std_H[self.id_list.index(str(id_H))]
        img_M /= self.std_M[self.id_list.index(str(id_M))]
        img_S /= self.std_S[self.id_list.index(str(id_S))]

        if self.phase == 'train':
            img_H = Image.fromarray(img_H)
            img_M = Image.fromarray(img_M)
            img_S = Image.fromarray(img_S)

            # Random Horizontal flip for Paired Images
            if random.random() > 0.5:
                img_S = transforms.functional.vflip(img_S)
            if random.random() > 0.5:
                img_M = transforms.functional.vflip(img_M)
            if random.random() > 0.5:
                img_H = transforms.functional.vflip(img_H)
            if random.random() > 0.5:
                img_S = transforms.functional.hflip(img_S)
            if random.random() > 0.5:
                img_M = transforms.functional.hflip(img_M)
            if random.random() > 0.5:
                img_H = transforms.functional.hflip(img_H)

            # i, j, h, w = transforms.RandomCrop.get_params(img_S, output_size=(self.img_crop_size, self.img_crop_size))
            # img_S = np.array(transforms.functional.crop(img_S, i, j, h, w))
            # img_M = np.array(transforms.functional.crop(img_M, i, j, h, w))
            # img_H = np.array(transforms.functional.crop(img_H, i, j, h, w))
            img_S = np.array(self.crop(img_S))
            img_M = np.array(self.crop(img_M))
            img_H = np.array(self.crop(img_H))

        img_S = img_S[np.newaxis, :, :]
        img_M = img_M[np.newaxis, :, :]
        img_H = img_H[np.newaxis, :, :]

        img_S = torch.tensor(img_S, dtype=torch.float32)
        img_M = torch.tensor(img_M, dtype=torch.float32)
        img_H = torch.tensor(img_H, dtype=torch.float32)

        return {'real_S': img_S, 'real_H':img_H, 'real_M':img_M,
                'id':[id_H,id_S,id_M], 'path_H':path_H, 'path_S':path_S, 'path_M':path_M}

    def __len__(self):
        data_size = min(len(self.path_H), len(self.path_M))
        data_size = min(data_size, len(self.path_S))
        return int(data_size)

class Dataset_test(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size

        self.path_H = []
        self.path_S = []
        dir_id_list_test = self.opt.data_dir + '{}_test'.format(self.opt.data_type)

        self.dir = sorted(glob(dir_id_list_test+'/*'))

        for p in self.dir:
            tmp_path_H = glob(p + '/H/*.mat')
            tmp_path_S = glob(p + '/S/*.mat')

            self.path_H.extend(tmp_path_H)
            self.path_S.extend(tmp_path_S)

        self.path_H = sorted(self.path_H)
        self.path_S = sorted(self.path_S)

    def __getitem__(self, index):

        path_H = self.path_H[index]
        path_S = self.path_S[index]

        img_H = _read_mat(path_H)
        img_S = _read_mat(path_S)

        # normalize for each volume
        m_H,v_H = np.mean(img_H), np.std(img_H)
        m_S,v_S = np.mean(img_S), np.std(img_S)
        img_H = (img_H-m_H)/v_H
        img_S = (img_S-m_S)/v_S

        img_S = img_S[np.newaxis, :, :]
        img_H = img_H[np.newaxis, :, :]

        img_S = torch.tensor(img_S, dtype=torch.float32)
        img_H = torch.tensor(img_H, dtype=torch.float32)

        return {'real_S': img_S, 'real_H':img_H, 'path_H':path_H, 'path_S':path_S,
                'stat_H':[m_H,v_H],'stat_S':[m_S,v_S],'id':[0]}

    def __len__(self):
        return int(len(self.path_H))

class Dataset_test_M(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size

        self.path_H = []
        self.path_S = []
        self.path_M = []
        dir_id_list_test = self.opt.data_dir + '{}_test'.format(self.opt.data_type)
        self.dir = sorted(glob(dir_id_list_test+'/*'))

        for p in self.dir:
            tmp_path_H = glob(p + '/Hr68/*.mat')
            tmp_path_S = glob(p + '/Hr40/*.mat')
            tmp_path_M = glob(p + '/Hr49/*.mat')

            self.path_H.extend(tmp_path_H)
            self.path_M.extend(tmp_path_M)
            self.path_S.extend(tmp_path_S)

        self.path_H = sorted(self.path_H)
        self.path_M = sorted(self.path_M)
        self.path_S = sorted(self.path_S)

    def __getitem__(self, index):

        path_H = self.path_H[index]
        path_S = self.path_S[index]
        path_M = self.path_M[index]

        img_H = _read_mat(path_H)
        img_M = _read_mat(path_M)
        img_S = _read_mat(path_S)

        # normalize for each volume
        m_H,v_H=np.mean(img_H), np.std(img_H)
        m_S,v_S=np.mean(img_S), np.std(img_S)
        m_M,v_M=np.mean(img_M), np.std(img_M)

        img_H = (img_H - m_H) / v_H
        img_S = (img_S - m_S) / v_S
        img_M = (img_M - m_M) / v_M

        img_S = img_S[np.newaxis, :, :]
        img_M = img_M[np.newaxis, :, :]
        img_H = img_H[np.newaxis, :, :]

        img_S = torch.tensor(img_S, dtype=torch.float32)
        img_M = torch.tensor(img_M, dtype=torch.float32)
        img_H = torch.tensor(img_H, dtype=torch.float32)

        return {'real_S': img_S, 'real_H':img_H, 'real_M':img_M,
                'path_H':path_H, 'path_S':path_S, 'path_M':path_M,
                'stat_H':[m_H,v_H], 'stat_S':[m_S,v_S], 'stat_M':[m_M,v_M], 'id':[0]}

    def __len__(self):
        data_size = min(len(self.path_H), len(self.path_M))
        data_size = min(data_size, len(self.path_S))
        return int(data_size)

def _read_dicom(path):
    tmp = pydicom.dcmread(path)
    pixel_value = np.array(tmp.pixel_array)

    HU_value = pixel_value * tmp.RescaleSlope + tmp.RescaleIntercept
    HU_value = _preprocessing(HU_value)
    return HU_value

def _read_mat(path):
    tmp = sio.loadmat(path)
    pixel_value = tmp['img']
    return pixel_value

def _preprocessing(image):
    output = np.copy(image)

    mu_h2o = 0.0192
    output = output * mu_h2o / 1000 + mu_h2o
    output[output < 0] = 0
    return output