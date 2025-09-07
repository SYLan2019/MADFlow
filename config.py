# device settings
device = 'cuda:0'# or 'cpu'
# data settings
dataset_dir = 'dataset_path' # parent directory of class folders

feature_dir = 'feature_dir' # directory where features are stored and loaded from
dataset = "mvtec3d"
depth_feature_dir = "depth_feature_dir"

use_3D_dataset = True # is MVTec 3D used?
pre_extracted = True # were feature pre-extracted with extract_features? (recommended)
modelname = "mvtec3d_test_1" # export evaluations/logs with this name
print(modelname)
# inputs
img_len = 768 # width/height of input image
img_size = (img_len, img_len)
img_dims = [3] + list(img_size)
depth_len = img_len // 4 # width/height of depth maps
size = depth_len
depth_downscale = 8 # to which factor depth maps are downsampled by unshuffling
depth_channels = depth_downscale ** 2 # channels per pixel after unshuffling
map_len = img_len // 32 # feature map width/height (dependent on feature extractor!)
extract_layer = [19,26,35] # layer from which features are extracted
img_feat_dims = 608 # number of image features (dependent on feature extractor!)

pc_semantic_dims = 1152

map_size = (img_size[0] // 32, img_size[1] // 32)
fc_internal = 1024

if not use_3D_dataset:
    mode = 'RGB' # force RGB if no 3D data is available
else:
    mode = ['RGB', 'depth', 'combi'][2]
n_feat = {'RGB': img_feat_dims, 'depth': depth_channels, 'combi': img_feat_dims + depth_channels}[mode]
n_feat_img = img_feat_dims + depth_channels
n_feat_depth = 64

training_mask = (mode != 'RGB') # use foreground mask for training?
eval_mask = (mode != 'RGB') # use foreground mask for evaluation?

# 3D settings
dilate_mask = True
dilate_size = 8
n_fills = 3
bg_thresh = 7e-3

# transformation settings
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
clamp = 1.9 # clamping parameter
n_coupling_blocks = 4 # higher = more flexible = more unstable
channels_hidden_teacher = 64 # number of neurons in hidden layers of internal networks
use_gamma = True
kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]
pos_enc = True # use positional encoding
pos_enc_dim = 32 # number of dimensions of positional encoding
asymmetric_student = True
n_st_blocks = 4 # number of residual blocks in student

# training parameters
lr = 0.0002 # learning rate
lr_depth = 2e-4 # learning rate
lr_rgb = 1e-3 # learning rate
batch_size = 16
eval_batch_size = 16

epochs = 200
train_epoch = 50

# output settings
verbose = True
hide_tqdm_bar = True
save_model = True

# eval settings
localize = True