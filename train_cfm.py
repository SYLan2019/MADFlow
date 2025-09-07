import random

from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from model_dual import *
from utils import *
import config as c1
from point_faeture_extractor import point_extractor
from clcloss import *


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
    model_d = FeatureProjectionConv(inchannel=1152, outchannel=608)
    model_d.to(c1.device)
    optimizer_d = torch.optim.Adam(params = model_d.parameters(), lr=1e-3, eps=1e-08, weight_decay=1e-5)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=15, gamma=0.9)

    mean_nll_obs = Score_Observer('AUROC mean over maps')
    max_nll_obs = Score_Observer('AUROC  max over maps')

    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    l2_metric = nn.MSELoss()

    max_auc = 0

    point_features = list()
    test_point_features = list()

    for data in tqdm(train_loader):
        depth, fg, labels, image, features, pcs = data
        depth_feature = point_extractor(pcs).cpu().numpy()
        point_features.append((depth_feature))
    print("training....")

    for data in tqdm(test_loader):
        depth, fg, labels, image, features, pcs, _ = data
        depth_feature = point_extractor(pcs).cpu().numpy()
        test_point_features.append((depth_feature))
    print("testing....")

    for epoch in range(c1.train_epoch):
        # train some epochs
        model_d.train()
        if c1.verbose:
            print(F'\nTrain epoch {epoch}')

        train_loss_d = 0

        for i, data in enumerate(train_loader):
            optimizer_d.zero_grad()

            depth, fg, labels, image, features, pcs = data

            depth = depth * scale

            depth, fg, labels, image, features = to_device([depth, fg, labels, image, features])
            fg = dilation(fg, c1.dilate_size) if c1.dilate_mask else fg

            depth_feature = point_features[i]
            depth_feature = torch.Tensor(depth_feature).to(c.device)

            features = torch.nn.functional.interpolate(features, size=(192, 192), mode='bilinear',
                                                       align_corners=False)

            z_pred = model_d(depth_feature)

            features = features.reshape(features.shape[0], features.shape[1], -1)
            features = features.permute(0, 2, 1)
            z_pred = z_pred.reshape(z_pred.shape[0], z_pred.shape[1], -1)
            z_pred = z_pred.permute(0, 2, 1)
            fg = fg.squeeze(1)
            fg = fg.reshape(fg.shape[0], -1)
            feature_mask = fg!=0

            loss_d = 1 - metric(features[feature_mask], z_pred[feature_mask]).mean()
            loss_d +=  l2_metric(features[feature_mask], z_pred[feature_mask])
            train_loss_d += (t2np(loss_d))
            loss_d.backward()
            optimizer_d.step()
        lr_schedule.step()

        mean_train_loss_d = train_loss_d / len(train_loader)
        print('Epoch: {:d}/{:d} \t depth train loss: {:.4f}'.format(epoch, c1.epochs, mean_train_loss_d))

        # evaluate
        if ((epoch+1) % 2 == 0):
            model_d.eval()
            if c1.verbose:
                print('\nCompute loss and scores on test set:')
            test_loss_d = list()
            test_labels = list()
            img_nll = list()
            max_nlls = list()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    depth, fg, labels, image, features, pcs, _ = data

                    depth = depth * scale

                    depth, fg, image, features = to_device([depth, fg, image, features])
                    fg = dilation(fg, c1.dilate_size) if c1.dilate_mask else fg

                    depth_feature = test_point_features[i]
                    depth_feature = torch.Tensor(depth_feature).to(c.device)

                    features = torch.nn.functional.interpolate(features, size=(192, 192), mode='bilinear',
                                                               align_corners=False)

                    z_pred = model_d(depth_feature)

                    features = features.reshape(features.shape[0], features.shape[1], -1)
                    features = features.permute(0, 2, 1)

                    z_pred = z_pred.reshape(z_pred.shape[0], z_pred.shape[1], -1)
                    z_pred = z_pred.permute(0, 2, 1)

                    fg = fg.squeeze(1)
                    fg = fg.reshape(fg.shape[0], -1)
                    feature_mask = fg != 0

                    ano = (torch.nn.functional.normalize(features, dim=2) - torch.nn.functional.normalize(
                            z_pred, dim=2)).pow(2).sum(2).sqrt()
                    ano[~feature_mask] = 0
                    ano = ano.reshape(-1, 1, 24, 24)

                    ano = ano.squeeze(1)
                    loss_d = torch.mean(ano, dim=(-1, -2))

                    img_nll.append(t2np(loss_d))
                    max_nlls.append(np.max(t2np(ano), axis=(-1, -2)))

                    test_loss_d.append(loss_d.mean().item())

                    test_labels.append(labels)

            test_loss_d = np.mean(np.array(test_loss_d))

            if c1.verbose:
                print('Epoch: {:d} \t depth test_loss: {:.4f}'.format(epoch, test_loss_d))

            img_nll = np.concatenate(img_nll, axis=0)
            max_nlls = np.concatenate(max_nlls, axis=0)
            test_labels = np.concatenate(test_labels)
            is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

            mean_flag, mean_score = mean_nll_obs.update(roc_auc_score(is_anomaly, img_nll), epoch,
                                                        print_score=c1.verbose)
            max_flag, max_score = max_nll_obs.update(roc_auc_score(is_anomaly, max_nlls), epoch,
                                                     print_score=c1.verbose)

            max_s = max(mean_score, max_score)
            if max_s >= max_auc:
                max_auc = max_s
                save_weights_cfm(model_d, 'cfm')

    return mean_nll_obs, max_nll_obs

if __name__ == "__main__":
    random_seed = 16523
    seed_everything(random_seed)

    train_dataset(train)