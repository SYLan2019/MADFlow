import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from os.path import join
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import config as c
import math
import tifffile as tiff

def train_dataset(train_function, get_mask=True):
    # MvTec3D
    all_classes = ['potato', 'foam', 'tire', 'carrot', 'cookie', 'cable_gland', 'bagel', 'peach', 'dowel', 'rope']

    # Eyecandies
    all_classes = ['CandyCane', 'HazelnutTruffle', 'GummyBear', 'Lollipop', 'ChocolatePraline', 'Confetto',
                   'ChocolateCookie', 'LicoriceSandwich', 'Marshmallow', 'PeppermintCandy']
    max_scores = list()
    pixel_scores = list()
    aupro30_scores = list()
    for i_c, cn in enumerate(all_classes):
        c.class_name = cn
        if cn == "tire":
            scale = 1
        else:
            scale = 100
        print('\n\nTrain class ' + c.class_name)

        train_set, test_set = load_datasets(get_mask=get_mask)
        train_loader, test_loader = make_dataloaders(train_set, test_set)
        max_sc, pixel_sc, aupro30_sc = train_function(train_loader, test_loader, scale)
        max_scores.append(max_sc)
        pixel_scores.append(pixel_sc)
        aupro30_scores.append(aupro30_sc)
        # break
    last_max = np.mean([s.last_score for s in max_scores])
    best_max = np.mean([s.best_score for s in max_scores])

    last_pixel = np.mean([s.last_score for s in pixel_scores])
    best_pixel = np.mean([s.best_score for s in pixel_scores])

    last_aupro30 = np.mean([s.last_score for s in aupro30_scores])
    best_aupro30 = np.mean([s.best_score for s in aupro30_scores])
    print('\nI-AUROC % after last epoch\n\t max over maps: {:.2f} \n P-AUROC % after last epoch:{:.2f}'.format(last_max, last_pixel))
    print('best I-AUROC %\n\t max over maps: {:.2f}\n best P-AUROC: {:.2f} '.format(best_max, best_pixel))

    print('\n AUPRO after last epoch\n\tAUPRO@30 over maps: {:.2f}'.format(last_aupro30))
    print('\n best AUPRO \n\tAUPRO@30 over maps: {:.2f}'.format(best_aupro30))

def dilation(map, size):
    map = t2np(map)
    kernel = np.ones([size, size])
    for i in range(len(map)):
        map[i, 0] = binary_dilation(map[i, 0], kernel)
    map = torch.FloatTensor(map).to(c.device)
    return map

def cal_topk(nll):
    top_k = int(nll.shape[1] * nll.shape[2] * c.topk)
    anomaly_score = np.mean(
        nll.reshape(nll.shape[0], -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
        axis=1)
    return anomaly_score


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def to_device(tensors, device=c.device):
    return [t.to(device) for t in tensors]


def get_st_loss(target, output, mask=None, per_sample=False, per_pixel=False,classes=None):
    if not c.training_mask:
        mask = 0 * mask + 1

    loss_per_pixel = torch.mean(mask * (target - output) ** 2, dim=1)

    if per_pixel:
        return loss_per_pixel

    loss_per_sample = torch.mean(loss_per_pixel, dim=(-1, -2))
    if per_sample:
        return loss_per_sample
    return loss_per_sample.mean()


def get_nf_loss(z, jac, mask=None, per_sample=False, per_pixel=False):
    if not c.training_mask:
        mask = 0 * mask + 1
    loss_per_pixel = (0.5 * torch.sum(mask * z ** 2, dim=1) - jac * mask[:, 0])
    if per_pixel:
        return loss_per_pixel
    loss_per_sample = torch.mean(loss_per_pixel, dim=(-1, -2))
    if per_sample:
        return loss_per_sample
    return loss_per_sample.mean()


def load_datasets(get_mask=True, get_features=c.pre_extracted):
    trainset = DefectDataset(set='train', get_mask=True, get_features=get_features)
    testset = DefectDataset(set='test', get_mask=get_mask, get_features=get_features)
    return trainset, testset


def load_img_datasets(dataset_dir, class_name):

    def target_transform(target):
        return class_perm[target]

    data_dir_train = os.path.join(dataset_dir, class_name, 'train')
    data_dir_test = os.path.join(dataset_dir, class_name, 'test')
    classes = os.listdir(data_dir_test)
    if 'good' not in classes:
        raise RuntimeError(
            'There should exist a subdirectory "good". Read the doc of this function for further information.')
    classes.sort()
    class_perm = list()
    class_idx = 1
    for cl in classes:
        if cl == 'good':
            class_perm.append(0)
        else:
            class_perm.append(class_idx)
            class_idx += 1

    image_transforms = transforms.Compose([transforms.Resize(c.img_size), transforms.ToTensor(),
                                           transforms.Normalize(c.norm_mean, c.norm_std)])
    valid_img = (lambda x: 'rgb' in x and x.endswith('png')) if c.use_3D_dataset else None
    trainset = ImageFolder(data_dir_train, transform=image_transforms, is_valid_file=valid_img)
    testset = ImageFolder(data_dir_test, transform=image_transforms, target_transform=target_transform,
                          is_valid_file=valid_img)
    return trainset, testset


def make_dataloaders(trainset, testset, shuffle_train=False, drop_last=False):
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=shuffle_train,
                                              drop_last=drop_last)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.eval_batch_size, shuffle=False,
                                             drop_last=False)
    return trainloader, testloader


def downsampling(x, size, to_tensor=False, bin=True):
    if to_tensor:
        x = torch.FloatTensor(x).to(c.device)
    down = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    if bin:
        down[down > 0] = 1
    return down


def organized_pc_to_unorganized_pc(organized_pc):
    # WxHxC ==> WH x C
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def unorganized_pc_to_organized_pc(unorganized_pc):
    # WHxC ==> W x H x C
    n = int(math.sqrt(unorganized_pc.shape[0]))
    return unorganized_pc.reshape(n, n, unorganized_pc.shape[1])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img


# 缩放图像
def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    # 1xCxWxH
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    # 1xCxW'xH'
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0).contiguous()
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).contiguous().numpy()



class DefectDataset(Dataset):
    def __init__(self, set='train', get_mask=True, get_features=True):
        super(DefectDataset, self).__init__()
        self.set = set
        self.labels = list()
        self.masks = list()
        self.images = list()
        self.depths = list()
        self.class_names = ['good']
        self.get_mask = get_mask
        self.get_features = get_features
        self.image_transforms = transforms.Compose([transforms.Resize(c.img_size), transforms.ToTensor(),
                                                    transforms.Normalize(c.norm_mean, c.norm_std)])
        root = join(c.dataset_dir, c.class_name)
        set_dir = os.path.join(root, set)

        subclass = os.listdir(set_dir)
        subclass.sort()
        class_counter = 1
        for sc in subclass:
            if sc == 'good':
                label = 0
            else:
                label = class_counter
                self.class_names.append(sc)
                class_counter += 1
            sub_dir = os.path.join(set_dir, sc)
            img_dir = join(sub_dir, 'rgb') if c.use_3D_dataset else sub_dir
            img_paths = os.listdir(img_dir)
            img_paths.sort()

            for p in img_paths:
                i_path = os.path.join(img_dir, p)
                if not i_path.lower().endswith(
                        ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                    continue
                self.images.append(i_path)
                self.labels.append(label)
                if self.set == 'test' and self.get_mask:
                    extension = '_mask' if sc != 'good' else ''
                    mask_path = i_path.replace('rgb', 'gt') if c.use_3D_dataset else os.path.join(root, 'ground_truth',
                                                                                                  sc,
                                                                                                  p[:-4] + extension + p[-4:])
                    self.masks.append(mask_path)
                if c.use_3D_dataset:
                    self.depths.append(i_path.replace('rgb', 'z')[:-4] + '.npy')

        if get_features:
            self.features = np.load(os.path.join(c.feature_dir, c.class_name, set + '.npy'))
            self.depth_features = np.load(os.path.join(c.depth_feature_dir, c.class_name, set + '.npy'))

        self.img_mean = torch.FloatTensor(c.norm_mean)[:, None, None]
        self.img_std = torch.FloatTensor(c.norm_std)[:, None, None]

    def __len__(self):
        return len(self.features)

    def transform(self, x, img_len, binary=False):
        x = x.copy()
        x = torch.FloatTensor(x)
        if len(x.shape) == 2:
            x = x[None, None]
            channels = 1
        elif len(x.shape) == 3:
            x = x.permute(2, 0, 1)[None]
            channels = x.shape[1]
        else:
            raise Exception(f'invalid dimensions of x:{x.shape}')

        x = downsampling(x, (img_len, img_len), bin=binary)
        x = x.reshape(channels, img_len, img_len)
        return x

    def get_3D(self, index):
        sample = np.load(self.depths[index])
        depth = sample[:, :, 0]
        fg = sample[:, :, -1]
        mean_fg = np.sum(fg * depth) / np.sum(fg)
        depth = fg * depth + (1 - fg) * mean_fg
        depth = (depth - mean_fg) * 1
        return depth, fg

    def __getitem__(self, index):
        if self.set == "train":
            label = 0
            feat = self.features[index] if self.get_features else 0
        else:
            label = self.labels[index]
            feat = self.features[index] if self.get_features else 0

        if c.use_3D_dataset:
            depth, fg = self.get_3D(index)
            depth = self.transform(depth, c.depth_len, binary=False)
            depth_feature = self.depth_features[index]

            fg = self.transform(fg, c.depth_len, binary=True)
        else:
            depth = torch.zeros([1, c.depth_len, c.depth_len])
            fg = torch.ones([1, c.depth_len, c.depth_len])

        if self.set == 'test' or not self.get_features:
            with open(self.images[index], 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = self.image_transforms(img)
        else:
            img = 0

        ret = [depth, fg, label, img, feat, depth_feature]

        if self.set == 'test' and self.get_mask:
            with open(self.masks[index], 'rb') as f:
                mask = Image.open(f)
                mask = self.transform(np.array(mask), c.depth_len, binary=True)[:1]
                mask[mask > 0] = 1
                ret.append(mask)
        return ret


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name, percentage=True):
        self.name = name
        self.max_epoch = 0
        self.best_score = None
        self.last_score = None
        self.percentage = percentage

    def update(self, score, epoch, print_score=False):
        if self.percentage:
            score = score * 100
        self.last_score = score
        improved = False
        if epoch == 3 or score > self.best_score:
            self.best_score = score
            improved = True
        if print_score:
            self.print_score()
        return improved,score

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t best: {:.2f}'.format(self.name, self.last_score, self.best_score))
