import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from os.path import join
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import datetime
import config as c
from model_dual import *
from utils_visulize import *
from visualize import *
from viz import *
from au_pro_util import calculate_au_pro
from sklearn.manifold import TSNE
from scipy import stats

def localize(image, depth, st_pixel, labels, fg, mask, batch_ind):
    for i in range(fg.shape[0]):
        if labels[i] > 0:
            fg_i = t2np(fg[i, 0])
            depth_viz = t2np(depth[i, 0])
            depth_viz[fg_i == 0] = np.nan
            viz_maps(t2np(image[i]), depth_viz, t2np(mask[i, 0]), t2np(st_pixel[i]), fg_i,
                     str(batch_ind) + '_' + str(i), norm=True)


def evaluate(test_loader, scale):
    model = Model(depth=False)
    model_d = FeatureProjectionConv(inchannel=1152, outchannel=608)

    model_d = load_weights_cfm(model_d, "cfm")
    model = load_weights(model,"flow")

    model.to(c.device)
    model_d.to(c.device)
    model.eval()
    model_d.eval()
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    test_loss = list()
    test_labels = list()
    img_nll = list()
    max_nlls = list()
    nll_list = list()
    img_list = list()

    score_maps = list()
    gt_masks = list()
    up = torch.nn.Upsample(size=None, scale_factor=c.depth_len // c.map_len, mode='bicubic',
                           align_corners=False)

    masks = list()

    au_score_maps = list()
    au_gt_masks = list()

    z_list = list()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            depth, fg, labels, image, features, depth_feature, mask, img_path, mask_path = data
            img_list.extend(img_path)

            depth = depth * scale

            depth, fg, image, features, depth_feature, mask = to_device(
                [depth, fg, image, features, depth_feature, mask])
            fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

            fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)

            ori_features = features.clone()
            features = torch.nn.functional.interpolate(features, size=(c.size, c.size), mode='bilinear',
                                                       align_corners=False)

            z_pred = model_d(depth_feature, features)

            ano = (torch.nn.functional.normalize(features, dim=1) - torch.nn.functional.normalize(z_pred,
                                                                                                  dim=1)).pow(
                2).sum(1).sqrt()
            ano = ano.unsqueeze(1)

            z, jac = model(ori_features, depth, ano, depth_feature, False)
            loss = get_nf_loss(z, jac, fg_down, per_sample=True)
            nll = get_nf_loss(z, jac, fg_down, per_pixel=True)
            z_list.append(t2np(nll.view(nll.shape[0], -1)))

            img_nll.append(t2np(loss))
            max_nlls.append(np.max(t2np(nll), axis=(-1, -2)))

            nll_list.append(t2np(nll))

            st_pixel = up((nll)[:, None])[:, 0]
            score_maps.append((t2np(st_pixel).flatten()))
            gt_masks.append(t2np(mask).flatten())

            au_score_maps.append(t2np(st_pixel))
            au_gt_masks.append(t2np(mask.squeeze(1)))

            masks.append(t2np(mask))

            test_loss.append(loss.mean().item())
            test_labels.append(labels)

    img_nll = np.concatenate(img_nll)
    max_nlls = np.concatenate(max_nlls)

    nll_list = np.concatenate(nll_list)

    score_maps = np.concatenate(score_maps)
    gt_masks = np.concatenate(gt_masks)
    masks = np.concatenate(masks)

    test_labels = np.concatenate(test_labels)

    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])


    mean_score = roc_auc_score(is_anomaly, img_nll)
    max_score = roc_auc_score(is_anomaly, max_nlls)
    pixel_score = roc_auc_score(gt_masks, score_maps)

    return mean_score, max_score, pixel_score, img_list, is_anomaly, masks, nll_list


if __name__ == "__main__":
    all_classes = [d for d in os.listdir(c.dataset_dir) if os.path.isdir(join(c.dataset_dir, d))]
    max_scores = list()
    mean_scores = list()
    pixel_scores = list()
    for i_c, cn in enumerate(all_classes):
        c.class_name = cn
        print('\nEvaluate class ' + c.class_name)
        train_set, test_set = load_datasets1(get_mask=True)
        _, test_loader = make_dataloaders(train_set, test_set)

        if cn == "tire":
            scale = 1
        else:
            scale = 100

        mean_sc, max_sc, pixel_sc, img_list, gt_label_list, gt_mask, nll_list = evaluate(test_loader, scale)
        mean_scores.append(mean_sc)
        max_scores.append(max_sc)
        pixel_scores.append(pixel_sc)


    mean_scores = np.mean(mean_scores) * 100
    max_scores = np.mean(max_scores) * 100
    pixel_scores = np.mean(pixel_scores) * 100
    print("I-AUROC % mean over map:{:.2f} \t max over map:{:.2f}".format(mean_scores, max_scores))
    print("P-AUROC % over map: {:.2f}".format(pixel_scores))