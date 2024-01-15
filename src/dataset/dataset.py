#!/usr/bin/python
# encoding: utf-8

import random
import torch
import math
import re
#from torch._C import float16
#from torch._C import uint8

from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
import bisect
import warnings
from PIL import Image
import numpy as np
import string
import torch.nn as nn

sys.path.append('../')
from utils import str_filt
from utils.labelmaps import get_vocabulary, labels2strs
from IPython import embed
from src.interfaces.TextSegmentation.lib.model_zoo.get_model import get_model
from src.interfaces.TextSegmentation.lib.cfg_helper import cfg_unique_holder as cfguh, \
    get_experiment_id, \
    experiment_folder, \
    common_initiates, \
    set_debug_cfg

import torch.nn.functional as F
from src.interfaces.TextSegmentation.configs.cfg_model import cfg_texrnet as cfg_mdel
from src.interfaces.TextSegmentation.configs.cfg_base import cfg_train, cfg_test

from src.interfaces.TextSegmentation.train_utils import \
    set_cfg as set_cfg_train, \
    set_cfg_hrnetw48 as set_cfg_hrnetw48_train, \
    ts, ts_with_classifier, train

from src.interfaces.TextSegmentation.eval_utils import \
    set_cfg as set_cfg_eval, \
    set_cfg_hrnetw48 as set_cfg_hrnetw48_eval, \
    es, eval
import copy
from sklearn.metrics import confusion_matrix
random.seed(0)
import cv2
#torch.multiprocessing.set_start_method('spawn')
scale = 0.90
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
def rand_crop(im):
    w, h = im.size
    p1 = (random.uniform(0, w*(1-scale)), random.uniform(0, h*(1-scale)))
    p2 = (p1[0] + scale*w, p1[1] + scale*h)
    return im.crop(p1 + p2)


def central_crop(im):
    w, h = im.size
    p1 = (((1-scale)*w/2), (1-scale)*h/2)
    p2 = ((1+scale)*w/2, (1+scale)*h/2)
    return im.crop(p1 + p2)


def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im
def buf2Tensorhr(txn, key):
    #import ipdb;ipdb.set_trace()
    imgbuf = txn.get(key)
    #print(key)
    #import ipdb;ipdb.set_trace()
    '''buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)'''
    #im = Image.open(buf)
    #img=F.pil_to_tensor(im)
    im=np.frombuffer(imgbuf,dtype=np.float32)
    #print(im)
    #print(im.shape)
    im1=im[12288:16384]
    im2=im[0:12288]
    '''img1 = cv2.imdecode(im1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(im2, cv2.IMREAD_COLOR)
    print((img1.shape))'''
    img0=torch.tensor(im)
    '''img3=transforms.ToTensor(im1)
    img4=transforms.ToTensor(im2)'''
    img=img0.view(4,32,128)
    #img=torch.cat((img3,img4),0)
    #print(im.shape)
    return img
def buf2Tensorlr(txn, key):
    #import ipdb;ipdb.set_trace()
    imgbuf = txn.get(key)
    #print(key)
    #import ipdb;ipdb.set_trace()
    '''buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)'''
    #im = Image.open(buf)
    #img=F.pil_to_tensor(im)
    im=np.frombuffer(imgbuf,dtype=np.float32)
    #print(im)
    #print(im.shape)
    im1=im[3072:4096]
    im2=im[0:3072]
    '''img1 = cv2.imdecode(im1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(im2, cv2.IMREAD_COLOR)
    img3=transforms.ToTensor(img1)
    img4=transforms.ToTensor(img2)'''
    img0=torch.tensor(im)
    img=img0.view(4,16,64)
    #print(im.shape)
    return img
class reversemask(object):
    def __init__(self) :
        super().__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __call__(self,imglr,imghr):#b,c,h,w tensor 4channel
        cfg1 = copy.deepcopy(cfg_test)
        cfg1.EXPERIMENT_ID = None
        cfg1.MODEL = copy.deepcopy(cfg_mdel)
        cfg1 = set_cfg_eval(cfg1, dsname='textseg')
        cfg1 = set_cfg_hrnetw48_eval(cfg1)
        cfg1.MODEL.TEXRNET.PRETRAINED_PTH = 'pth/texrnet_hrnet.pth.tar'
        exec_es = es()
        tester = eval(cfg1)
        tester.register_stage(exec_es)
        model=tester(0)
        model =model.to(self.device)
        model = torch.nn.DataParallel(model, device_ids=range(2))
        #import ipdb;ipdb.set_trace()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        img1=F.interpolate(imghr,(4*imghr.size(2),4*imghr.size(3)),mode='bilinear',align_corners=True)
        segmask=torch.argmax((model(img1[ :,:3, :, :])['predrfn']),dim=1).unsqueeze(1)#单通道分割mask b,1,h,w
        for i in range(imghr.size(0)):
            cm0=confusion_matrix(np.array(segmask[i].cpu()).flatten(),np.array(imghr[i][3:4,:,:].cpu()).flatten())
            miou=np.diag(cm0) / (cm0.sum(axis=1) + cm0.sum(axis=0) - np.diag(cm0))#计算miou
            hrmean=np.array(segmask[i].cpu()).mean()*255.0#分割出的掩码平均值
            if hrmean>=35.0:#分割效果足够明显
                if (miou[0]<=0.35)&(miou[1]<=0.35):#传统方法的分割不正确
                    imagehr_maskr=np.array(imghr[i][3:4,:,:].cpu())
                    imagehr_maskr[np.where(imagehr_maskr==0)]=2
                    imagehr_maskr[np.where(imagehr_maskr==1)]=0
                    imagehr_maskr[np.where(imagehr_maskr==2)]=1
                    imghr[i][3:4,:,:]= torch.from_numpy(imagehr_maskr)#对mask做反转
                    imagelr_maskr=np.array(imglr[i][3:4,:,:].cpu())
                    imagelr_maskr[np.where(imagelr_maskr==0)]=2
                    imagelr_maskr[np.where(imagelr_maskr==1)]=0
                    imagelr_maskr[np.where(imagelr_maskr==2)]=1
                    imglr[i][3:4,:,:]= torch.from_numpy(imagelr_maskr)#对mask做反转
        return imglr,imghr
class lmdbDataset(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=31, test=True):
        super(lmdbDataset, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples

        self.max_len = max_len
        self.voc_type = voc_type

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)

        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())

        try:
            img = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
        except TypeError:
            img = buf2PIL(txn, b'image-%09d' % index, 'RGB')
        except IOError or len(label) > self.max_len:
            return self[index + 1]

        label_str = str_filt(word, self.voc_type)
        return img, label_str

class lmdbDataset_regular(Dataset):
    def __init__(self, root):
    
        self.root = root
        #self.opt = opt
        
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        

        self.image_path_list = []
        self.image_label_list = []
        # self.image_target_list = []

        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        #import ipdb;ipdb.set_trace()
        with self.env.begin(write=False) as txn:
            #import ipdb;ipdb.set_trace()
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            
            
            #if self.opt.data_filtering_off:
                # for fast check with no filtering
            #self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            
                # Filtering
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')

                if len(label) > 100:#self.opt.batch_max_length:
                    # print(f'The length of the label is longer than max_length: length
                    # {len(label)}, {label} in dataset {self.root}')
                    continue

                # By default, images containing characters which are not in opt.character are filtered.
                # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                '''out_of_char = f'[^{self.opt.character}]'
                if re.search(out_of_char, label.lower()):
                    continue'''

                
                label = label.lower()

                self.filtered_index_list.append(index)
                
                self.image_path_list.append('image-%09d.png' % index)
                #self.image_path_list.append('image-%09d.png' % index)
                self.image_label_list.append(label)

            self.nSamples = len(self.filtered_index_list)


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        key = self.filtered_index_list[index]
        path = self.image_path_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % key
            label = txn.get(label_key).decode('utf-8')
            
            img_key = 'image-%09d'.encode() % key
            imgbuf = txn.get(img_key)
            # assert self.image_label_list[index]==label if self.opt.sensitive else label.lower()
            # if self.image_label_list[index]!=label.lower():
            #     print(self.image_label_list[index])
            #     print(label.lower())
            #     print('--------error-------------')

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            #import ipdb;ipdb.set_trace()
            try:
                img_rgb = Image.open(buf).convert('RGB')  # for color image
                img_gray = img_rgb.convert('L')
            except IOError:
                # make dummy image and dummy label for corrupted image.
                print(f'Corrupted image for {path}')
                img_rgb = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                img_gray = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'
                
            #if not self.opt.sensitive:
            label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            '''out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)'''

        return [img_rgb, img_gray, label, path]
class lmdbDataset_real(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_real, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        
        try:
            '''img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')'''
            img_HR = buf2Tensorhr(txn, img_HR_key)
            img_lr = buf2Tensorlr(txn, img_lr_key)
            
            
        except IOError or len(word) > self.max_len:
            return self[index + 1]
        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str


class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mask = mask
    def __call__(self, img):
        #import ipdb;ipdb.set_trace()
        img = img.resize(self.size, self.interpolation)
        img_tensor = self.toTensor(img)
        
        if self.mask:
            #import ipdb;ipdb.set_trace()
            mask = img.convert('L')
            thres = np.array(mask).mean()
            mask = mask.point(lambda x: 0 if x > thres else 255)
            mask = self.toTensor(mask)
            img_tensor = torch.cat((img_tensor, mask), 0)   
        return img_tensor


class lmdbDataset_mix(Dataset):
    def __init__(self, root=None, voc_type='upper', max_len=100, test=False):
        super(lmdbDataset_mix, self).__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        word = str(txn.get(label_key).decode())
        if self.test:
            try:
                img_HR = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
                img_lr = buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            except:
                img_HR = buf2PIL(txn, b'image-%09d' % index, 'RGB')
                img_lr = img_HR

        else:
            img_HR = buf2PIL(txn, b'image_hr-%09d' % index, 'RGB')
            if random.uniform(0, 1) < 0.5:
                img_lr = buf2PIL(txn, b'image_lr-%09d' % index, 'RGB')
            else:
                img_lr = img_HR

        label_str = str_filt(word, self.voc_type)
        return img_HR, img_lr, label_str


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples
class NormalizePAD(object):
    
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img
class ResizeNormalize_regular(object):
    
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img
class alignCollate_regular(object):
    
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, img_grays, labels, path = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            transform = NormalizePAD((1, self.imgH, resized_max_w))

            resized_images = []
            resized_gray_images = []
            for image, img_gray in zip(images, img_grays):
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)
                
                resize_gray_image = img_gray.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_gray_images.append(transform(resize_gray_image))
                #import ipdb;ipdb.set_trace()
                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)
            
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
            image_gray_tensors = torch.cat([t.unsqueeze(0) for t in resized_gray_images], 0)
        else:
            transform = ResizeNormalize_regular((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
            
            image_gray_tensors = [transform(image) for image in img_grays]
            image_gray_tensors = torch.cat([t.unsqueeze(0) for t in image_gray_tensors], 0)

        return image_tensors, image_gray_tensors, labels, path
class alignCollate_syn(object):
    def __init__(self, imgH=64, imgW=256, down_sample_scale=4, keep_ratio=False, min_ratio=1, mask=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask

    def __call__(self, batch):
        images, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)

        images_hr = [transform(image) for image in images]
        images_hr = torch.cat([t.unsqueeze(0) for t in images_hr], 0)

        images_lr = [image.resize((image.size[0]//self.down_sample_scale, image.size[1]//self.down_sample_scale), Image.BICUBIC) for image in images]
        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)

        return images_hr, images_lr, label_strs


class alignCollate_real(alignCollate_syn):
    def __call__(self, batch):
        images_HR, images_lr, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH), self.mask)
        transform2 = resizeNormalize((imgW // self.down_sample_scale, imgH // self.down_sample_scale), self.mask)
        images_HR = [transform(image) for image in images_HR]
        images_HR = torch.cat([t.unsqueeze(0) for t in images_HR], 0)

        images_lr = [transform2(image) for image in images_lr]
        images_lr = torch.cat([t.unsqueeze(0) for t in images_lr], 0)
        

        return images_HR, images_lr, label_strs


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


if __name__ == '__main__':
    embed(header='dataset.py')