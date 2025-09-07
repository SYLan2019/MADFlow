import random

import numpy as np
import torch
from os.path import join
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import datetime
import config as c
from model_dual import *
from utils_both import *
from au_pro_util import calculate_au_pro


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(train_loader, test_loader, scale):
    model = Model(depth = False)
    model_d = FeatureProjectionConv(inchannel=1152, outchannel=608)

    model_d = load_weights_cfm(model_d, "cfm")

    model.to(c.device)
    model_d.to(c.device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-3, eps=1e-08, weight_decay=1e-5)
    lrscheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.85)

    max_nll_obs = Score_Observer('Image AUROC  max over maps')
    pixel_obs = Score_Observer('Pixel AUROC over maps')
    aupro30 = Score_Observer('AUPRO@30% over maps')

    for epoch in range(c.epochs):
        # train some epochs
        model.train()
        model_d.eval()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')

        train_loss = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            depth, fg, labels, image, features, depth_feature = data

            depth = depth * scale

            depth, fg, labels, image, features, depth_feature = to_device([depth, fg, labels, image, features, depth_feature])
            fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

            fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)

            # resize
            ori_features = features.clone()
            features = torch.nn.functional.interpolate(features, size=(192, 192), mode='bilinear',
                                                       align_corners=False)

            z_pred = model_d(depth_feature, features)
            ano = (torch.nn.functional.normalize(features,dim=1) - torch.nn.functional.normalize(z_pred,dim=1)).pow(2).sum(1).sqrt()
            ano = ano.unsqueeze(1)

            z, jac = model(ori_features, depth, ano.detach(), depth_feature, False)

            loss = get_nf_loss(z, jac, fg_down)
            train_loss += (t2np(loss))

            loss.backward()
            optimizer.step()

        lrscheduler.step()


        mean_train_loss = train_loss / len(train_loader)
        print('Epoch: {:d}/{:d} \t image train loss: {:.4f}'.format(epoch, c.epochs, mean_train_loss))

        # evaluate
        if ((epoch+1) % 4 == 0):
            model.eval()
            model_d.eval()
            if c.verbose:
                print('\nCompute loss and scores on test set:')
            test_loss = list()
            test_labels = list()
            max_nlls = list()

            score_maps = list()
            gt_masks = list()

            au_gt_masks = list()
            au_score_maps = list()

            up = torch.nn.Upsample(size=None, scale_factor=c.depth_len // c.map_len, mode='bicubic',
                                   align_corners=False)

            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    depth, fg, labels, image, features, depth_feature, mask = data

                    depth = depth * scale

                    depth, fg, image, features, depth_feature, mask = to_device([depth, fg, image, features, depth_feature, mask])
                    fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

                    fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)

                    ori_features = features.clone()
                    features = torch.nn.functional.interpolate(features, size=(192, 192), mode='bilinear',
                                                               align_corners=False)

                    z_pred = model_d(depth_feature, features)

                    ano = (torch.nn.functional.normalize(features, dim=1) - torch.nn.functional.normalize(z_pred,
                                                                                                          dim=1)).pow(
                        2).sum(1).sqrt()

                    ano = ano.unsqueeze(1)

                    z, jac = model(ori_features, depth, ano, depth_feature, False)
                    loss = get_nf_loss(z, jac, fg_down, per_sample=True)
                    nll = get_nf_loss(z, jac, fg_down, per_pixel=True)
                    max_nlls.append(np.max(t2np(nll), axis=(-1, -2)))

                    st_pixel = up((nll)[:, None])[:, 0]
                    score_maps.append((t2np(st_pixel).flatten()))
                    gt_masks.append(t2np(mask).flatten())

                    test_loss.append(loss.mean().item())

                    test_labels.append(labels)

            max_nlls = np.concatenate(max_nlls)

            test_loss = np.mean(np.array(test_loss))

            score_maps = np.concatenate(score_maps)
            gt_masks = np.concatenate(gt_masks)

            if c.verbose:
                print('Epoch: {:d} \t image test_loss: {:.4f}'.format(epoch, test_loss))

            test_labels = np.concatenate(test_labels)
            is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

            max_flag, max_score = max_nll_obs.update(roc_auc_score(is_anomaly, max_nlls), epoch,
                                                     print_score=c.verbose)

            pixel_flag, pixel_score = pixel_obs.update(roc_auc_score(gt_masks, score_maps), epoch,
                                                       print_score=c.verbose)


    return max_nll_obs, pixel_obs


if __name__ == "__main__":
    train_dataset(train)
