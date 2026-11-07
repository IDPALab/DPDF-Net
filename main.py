import os
import argparse
import time
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import scipy.ndimage as ndimage
from scipy.io import loadmat, savemat
from sklearn.decomposition import PCA
from models.DPDF_Net import DPDF_Net
from utils.data_processing import (position_train_and_test_point,mirror_hsi,train_and_test_data,train_and_test_label)
from utils.data_preparation import HSI_Dataset
from utils.metrics import output_metric
from utils.result_store_draw import draw_result_visualization, store_result
from utils.utils import load_wetland_data
from train import train_epoch, valid_epoch
from test import test_epoch
from thop import profile
from colormap.wetland_colormap import get_wetland_colormap

from losses import PrincipledSparsityLoss

# =========================================================
# Parameter Settings
# =========================================================
parser = argparse.ArgumentParser(description='DPDF_Net implementation')
parser.add_argument('-mt', '--model_type', type=str, default="DPDF_Net")
parser.add_argument('-crm', '--cnn_rnn_mode', type=str, default="CNN_3D_Classifier_1D")
parser.add_argument('-tm', '--transformer_mode', type=str, default="CAF")
parser.add_argument('-dpdf', '--dpdf_net_mode', type=str, default="3D")
parser.add_argument('-p', '--patch', type=int, default=7, help="patch size")
parser.add_argument('-bp', '--band_patch', type=int, default=3)
parser.add_argument('-e', '--epoch', type=int, default=10)
parser.add_argument('-tf', '--test_freq', type=int, default=20)
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-d','--dataset',type=str,default="MongCai",choices=['MongCai','CamPha','NC16','Botswana'])
parser.add_argument('-sm', '--sample_mode', type=str, default="fixed")
parser.add_argument('-s', '--sample_num', type=float, default=None)
parser.add_argument('-ls', '--lambda_sparse', type=float, default=0.1, help="weight for sparsity loss")
parser.add_argument('-kr', '--k_ratio', type=float, default=0.7, help="keep ratio for DRFCS (0.0~1.0)")

args = parser.parse_args()

# =========================================================

model_type = args.model_type
wetland_name = args.dataset
epoch = args.epoch
batch_size = args.batch_size
learning_rate = args.learning_rate
patches = args.patch
lambda_sparse = args.lambda_sparse
k_ratio = args.k_ratio


if wetland_name in ["MongCai", "CamPha"]:
    sample_mode = "fixed"
    sample_value = 200
elif wetland_name == "Botswana":
    sample_mode = "fixed"
    sample_value = 10
elif wetland_name == "NC16":
    sample_mode = "percentage"
    sample_value = 0.05
else:
    sample_mode = args.sample_mode
    sample_value = args.sample_num


time_now = time.localtime()
time_folder = (
    f"./logs/{time.strftime('%Y-%m-%d-%H-%M-%S', time_now)}"
    f"-{model_type}-wetland_{wetland_name}"
)

os.makedirs(time_folder, exist_ok=True)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = True
cudnn.benchmark = False

if wetland_name == "Botswana":
    image_path = f"./data/{wetland_name}.mat"
    label_path = f"./data/{wetland_name}_gt.mat"
else:
    image_path = f"./data/{wetland_name}_image.mat"
    label_path = f"./data/{wetland_name}_label.mat"

label_mat = loadmat(label_path)

if wetland_name == "HHK" or wetland_name == "ZY_YC":
    all_label_temp = label_mat['DataClass']
elif wetland_name == "Botswana":
    all_label_temp = label_mat['Botswana_gt']
else:
    all_label_temp = label_mat[f"{wetland_name}_label"]

num_classes = int(np.max(all_label_temp))
num_classes_actual = num_classes + 1

if wetland_name == "NC16":

    colors_np = np.array([
        [255, 255, 255],   # 0  Background
        [255, 192, 255],   # 1
        [211, 211, 211],   # 2
        [128, 128, 128],   # 3
        [176, 224, 230],   # 4
        [192, 192, 192],   # 5
        [235, 202, 121],   # 6
        [255,  99,  71],   # 7
        [127, 255, 212],   # 8
        [220,  20,  60],   # 9
        [ 34, 139,  34],   # 10
        [154, 205,  50],   # 11
        [  0, 255, 255],   # 12
        [255, 255,   0],   # 13
        [255, 127,  80],   # 14
        [189, 183, 107],   # 15
        [ 85, 107,  47]    # 16
    ], dtype=np.float32) / 255.0
else:

    try:
        colors_np = get_wetland_colormap(wetland_name)
    except Exception:
        colormap_mat = loadmat("./data/wetland_colormap.mat")

        colors_np = colormap_mat.get(f"colormap_{wetland_name}", None)


if colors_np is None:
    raise ValueError(f"Custom colormap not found for {wetland_name}.")
colors_np = np.asarray(colors_np, dtype=np.float32)

if colors_np.ndim != 2 or colors_np.shape[1] != 3:
    raise ValueError(f"The {wetland_name} colormap must be an N-by-3 RGB array. " f"Actual shape: {colors_np.shape}.")

if colors_np.shape[0] != num_classes_actual:
    raise ValueError(
        f"Incorrect number of colors in the {wetland_name} colormap: "
        f"found {colors_np.shape[0]}, "
        f"but the dataset requires {num_classes_actual}."
    )

if np.any(colors_np < 0) or np.any(colors_np > 1):
    raise ValueError(f"The {wetland_name} colormap values must be in [0, 1].")

colors_full = colors_np.copy()

save_colormap = mpl.colors.ListedColormap(colors_full, name=f"{wetland_name}_custom")
save_norm = mpl.colors.BoundaryNorm(np.arange(-0.5, num_classes_actual + 0.5, 1), save_colormap.N)

input, train_label_original, test_label_original = load_wetland_data(
    time_folder,
    image_path,
    label_path,
    wetland_name,
    sample_mode,
    sample_value,
    save_colormap
)

if wetland_name == "NC16":

    split_path = "./data/NC16_disjoint_split.mat"

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"NC16 spatially disjoint split file not found: {split_path}\n" f"Run split.py to generate the file.")

    nc16_split = loadmat(split_path)

    train_label = nc16_split['train_label']
    val_label = nc16_split['val_label']
    test_label = nc16_split['test_label']

else:

    train_label = np.zeros_like(train_label_original)
    val_label = np.zeros_like(train_label_original)
    rng = np.random.default_rng(SEED)

    for c in range(1, num_classes + 1):
        class_positions = np.argwhere(train_label_original == c)

        n_samples = len(class_positions)

        if n_samples == 0:
            continue

        if n_samples == 1:
            n_val = 0
        else:
            n_val = max(1, int(round(n_samples * 0.2)))

            n_val = min(n_val, n_samples - 1)

        shuffled_indices = rng.permutation(n_samples)

        val_indices = shuffled_indices[:n_val]

        train_indices = shuffled_indices[n_val:]

        for idx in train_indices:
            r, col = class_positions[idx]

            train_label[r, col] = c

        for idx in val_indices:
            r, col = class_positions[idx]

            val_label[r, col] = c


    test_label = test_label_original.copy()
print(
    f"[{wetland_name}] Sample counts | "
    f"Train: {np.count_nonzero(train_label > 0)} | "
    f"Validation: {np.count_nonzero(val_label > 0)} | "
    f"Test: {np.count_nonzero(test_label > 0)}"
)

all_label = (train_label + val_label + test_label)

background_count = np.sum(all_label == 0)

invalid_labels = all_label[(all_label < 0) | (all_label > num_classes)]

if invalid_labels.size > 0:
    raise ValueError(
        f"{wetland_name} contains invalid labels. "
        f"Expected range: 0 to {num_classes}. "
        f"Number of invalid labels: "
        f"{invalid_labels.size}"
    )

if wetland_name == "Botswana":
    if background_count == 0:
        raise ValueError("No background pixels (label 0) were found in all_label for Botswana.\n" "Check how background labels are handled during data loading and splitting.")

preprocessing_start = time.time()

if wetland_name in ["Botswana", "NC16"]:
    H, W, B = input.shape

    if wetland_name == "Botswana":
        pca_components = 30
    elif wetland_name == "NC16":
        pca_components = 50
    else:
        pca_components = B

    if B > pca_components:

        train_coords = np.argwhere(train_label > 0)

        if len(train_coords) == 0:
            raise RuntimeError(f"The {wetland_name} training set is empty. " f"Cannot fit PCA.")

        train_pixels = input[train_coords[:, 0], train_coords[:, 1],:]

        print("=" * 70)
        print(f"[{wetland_name}] PCA")
        print(f"Original bands       : {B}")
        print(f"PCA components       : {pca_components}")
        print(f"PCA fitting samples  : " f"{len(train_pixels)}")
        print("PCA fitting source   : Train samples only")
        print("=" * 70)

        pca = PCA(n_components=pca_components)
        pca.fit(train_pixels)

        input_2d = input.reshape(-1, B)
        input = pca.transform(input_2d).reshape(H, W, pca_components)


height, width, band = input.shape
input_normalize = np.zeros(input.shape, dtype=np.float32)
train_coords = np.argwhere(train_label > 0)

if len(train_coords) == 0:
    raise RuntimeError(f"The {wetland_name} training set is empty. " f"Cannot normalize the data.")

train_pixels = input[train_coords[:, 0], train_coords[:, 1],:]
train_mean = np.mean(train_pixels, axis=0)
train_std = np.std(train_pixels, axis=0)
train_std = np.maximum(train_std, 1e-8)
input_normalize = (input - train_mean.reshape(1, 1, -1)) / train_std.reshape(1, 1, -1)
input_normalize = input_normalize.astype(np.float32)

print("=" * 70)
print(f"[{wetland_name}] Normalization")
print("Normalization method : " "Standardization")
print("Statistics source     : " "Train samples only")
print(f"Train samples used    : " f"{len(train_pixels)}")
print("=" * 70)


if wetland_name == "Botswana":
    pseudo_bands = [min(20, band - 1), min(15, band - 1), min(10, band - 1)]
elif wetland_name == "NC16":
    pseudo_bands = [min(40, band - 1), min(20, band - 1), min(10, band - 1)]
elif wetland_name in ["MongCai", "CamPha", "ZY_YC", "HHK"]:
    b1 = min(60, band - 1)
    b2 = min(30, band - 1)
    b3 = min(10, band - 1)

    pseudo_bands = [b1, b2, b3]
else:
    pseudo_bands = [band // 2, band // 4, band // 8]

pseudo_color_img = input_normalize[:,:, pseudo_bands].copy()
pseudo_min = pseudo_color_img.min(axis=(0, 1), keepdims=True)
pseudo_max = pseudo_color_img.max(axis=(0, 1), keepdims=True)
pseudo_color_img = (pseudo_color_img - pseudo_min) / (pseudo_max - pseudo_min + 1e-8)
pseudo_color_img = np.clip(pseudo_color_img, 0, 1).astype(np.float32)
plt.imsave(f"{time_folder}/pseudo_color.png", pseudo_color_img, dpi=300)


if model_type == "DPDF_Net":
    model = DPDF_Net(num_classes=num_classes + 1, k_ratio=k_ratio)

mirror_image = mirror_hsi(height, width, band, input_normalize, patch=patches)
mirror_train_label = mirror_hsi(
    height,
    width,
    1,
    np.expand_dims(
        train_label,
        axis=2
    ),
    patch=patches
).reshape(
    mirror_image.shape[0],
    mirror_image.shape[1]
)

mirror_val_label = mirror_hsi(
    height,
    width,
    1,
    np.expand_dims(
        val_label,
        axis=2
    ),
    patch=patches
).reshape(
    mirror_image.shape[0],
    mirror_image.shape[1]
)

mirror_test_label = mirror_hsi(
    height,
    width,
    1,
    np.expand_dims(
        test_label,
        axis=2
    ),
    patch=patches
).reshape(
    mirror_image.shape[0],
    mirror_image.shape[1]
)


train_dataset = HSI_Dataset(mirror_image, mirror_train_label, True, patches)
val_dataset = HSI_Dataset(mirror_image, mirror_val_label, True, patches)
test_dataset = HSI_Dataset(mirror_image, mirror_test_label, True, patches)
train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
true_dataset = HSI_Dataset(mirror_image, mirror_test_label, False, patches)
true_loader = Data.DataLoader(true_dataset, batch_size=batch_size, shuffle=False)
preprocessing_time = (time.time() - preprocessing_start)

total_pos_true = true_dataset.indices


model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
criterion_sparse = PrincipledSparsityLoss(rho=k_ratio).cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epoch // 10), gamma=0.9)
dummy_input = torch.randn(1, 1, band, patches, patches).cuda()
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
flops_m = flops / 1e6
params_m = params / 1e6

print("=" * 70)
print("[Model Complexity]")
print(f"Params  : " f"{params_m:.4f} M")
print(f"FLOPs   : " f"{flops_m:.4f} M")
print(f"Target Keep Ratio : " f"{k_ratio:.2f}")
print("=" * 70)


print("=" * 70)
print(f"[START] Training " f"{model_type} " f"on {wetland_name}")
print(f"Batch Size: {batch_size} | " f"LR: {learning_rate} | " f"Lambda_S: {lambda_sparse}")
print("=" * 70)

tic = time.time()
epoch_result = np.zeros([3, epoch])

best_val_OA = -1.0
best_val_AA = 0.0
best_val_Kappa = 0.0
best_val_CA = None
best_val_CM = None
best_epoch = 0

best_model_path = (f"{time_folder}/best_model_state_dict.pkl")

for e in range(epoch):
    model.train()

    train_acc, train_loss, _, _ = train_epoch(
        model,
        train_loader,
        criterion,
        criterion_sparse,
        lambda_sparse,
        optimizer,
        e,
        epoch
    )

    scheduler.step()

    epoch_result[0][e] = e + 1
    epoch_result[1][e] = train_loss
    epoch_result[2][e] = train_acc


    if ((e + 1) % args.test_freq == 0 or e == epoch - 1):
        model.eval()

        with torch.no_grad():
            label_v, prediction_v = valid_epoch(model, val_loader, criterion, optimizer)

        (OA_v, AA_v, Kappa_v, CA_v, CM_v) = output_metric(label_v, prediction_v)

        print(
            f"Epoch: {e + 1:03d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val_OA: {OA_v:.4f} | "
            f"Val_AA: {AA_v:.4f} | "
            f"Val_Kappa: {Kappa_v:.4f}"
        )


        if OA_v > best_val_OA:
            best_val_OA = OA_v
            best_val_AA = AA_v
            best_val_Kappa = Kappa_v
            best_val_CA = CA_v
            best_val_CM = CM_v
            best_epoch = e + 1

            torch.save(model.state_dict(), best_model_path)

            print(f">>> Best model updated " f"(Epoch {best_epoch}, " f"Val OA = {best_val_OA:.4f})")


toc = time.time()

print("=" * 70)
print(f"Training Complete. " f"Total Time: {toc - tic:.2f}s")
print("=" * 70)

draw_result_visualization(time_folder, epoch_result)

if not os.path.exists(best_model_path):
    raise FileNotFoundError("The best model checkpoint was not found. " "Check that validation completed successfully.")

model.load_state_dict(torch.load(best_model_path, map_location='cuda'))

model.eval()

with torch.no_grad():
    label_test, prediction_test = valid_epoch(model, test_loader, criterion, optimizer)

(OA_test, AA_test, Kappa_test, CA_test, CM_test) = output_metric(label_test, prediction_test)

model.reset_sparsity_stats()
model.enable_sparsity_stats()

end_to_end_start = time.time()
model.eval()

with torch.no_grad():
    (
        pre_u,
        inference_time,
        per_patch_latency,
        peak_gpu_memory,
        total_patches
    ) = test_epoch(
        model,
        true_loader
    )

model.disable_sparsity_stats()

actual_sparsity = (model.get_actual_sparsity())


padding = patches // 2
prediction = np.zeros((height, width), dtype=float)
prediction_temp = np.zeros((height + 2 * padding, width + 2 * padding), dtype=float)

for i in range(total_pos_true.shape[0]):
    prediction_temp[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i]

for i in range(height):
    for j in range(width):
        prediction[i, j] = prediction_temp[padding + i, padding + j]


prediction_smooth = ndimage.median_filter(prediction, size=3)
mask = (all_label != 0)
prediction_masked = np.zeros_like(prediction_smooth)
prediction_masked[mask] = (prediction_smooth[mask])

if wetland_name == "HHK":
    prediction_for_show = (ndimage.maximum_filter(prediction_masked, size=9))
else:
    prediction_for_show = (prediction_masked)


background_mask = (all_label == 0)
prediction_for_show = (prediction_for_show.astype(np.int32))
prediction_smooth_for_show = (prediction_smooth.astype(np.int32))
prediction_for_show[background_mask] = 0
prediction_smooth_for_show[background_mask] = 0


if np.any(prediction_for_show < 0) or np.any(prediction_for_show > num_classes):
    invalid_prediction = prediction_for_show[
        (prediction_for_show < 0) |
        (prediction_for_show > num_classes)
    ]

    raise ValueError(
        f"{wetland_name} prediction_for_show "
        f"contains invalid class IDs. "
        f"Valid range: 0 to {num_classes}. "
        f"Number of invalid predictions: {invalid_prediction.size}."
    )


if not np.all(prediction_for_show[background_mask] == 0):
    raise ValueError("Background pixels still have nonzero prediction labels.")
if not np.allclose(colors_full[0], [1.0, 1.0, 1.0]):
    raise ValueError(f"The {wetland_name} background color must be white. " f"Actual color: {colors_full[0]}.")
prediction_rgb = colors_full[prediction_for_show]
prediction_smooth_rgb = colors_full[prediction_smooth_for_show]
ground_truth_rgb = colors_full[all_label.astype(np.int32)]

prediction_rgb[background_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
prediction_smooth_rgb[background_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
ground_truth_rgb[background_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)


plt.imsave(f"{time_folder}/prediction.png", prediction_rgb, dpi=300)
plt.imsave(f"{time_folder}/prediction_full_smooth.png", prediction_smooth_rgb, dpi=300)
plt.imsave(f"{time_folder}/ground_truth.png", ground_truth_rgb, dpi=300)

plt.imsave(f"{time_folder}/prediction.png", prediction_rgb, dpi=300)
plt.imsave(f"{time_folder}/prediction_full_smooth.png", prediction_smooth_rgb, dpi=300)
plt.imsave(f"{time_folder}/ground_truth.png", ground_truth_rgb, dpi=300)
savemat(f"{time_folder}/prediction_label.mat", {"prediction": prediction_smooth, "label": all_label})
end_to_end_cost = (time.time() - end_to_end_start)


store_result(
    time_folder,
    OA_test,
    AA_test,
    Kappa_test,
    CM_test,
    model_type,
    args.dpdf_net_mode,
    epoch,
    batch_size,
    patches,
    args.band_patch,
    learning_rate,
    args.weight_decay,
    0.9,
    sample_mode,
    sample_value
)


print("\n" + " " * 15 + "FINAL CLASSIFICATION REPORT")
print("=" * 70)
print(f"{'Dataset:':<30}" f"{wetland_name:>15}")

if wetland_name in ["MongCai", "CamPha"]:
    print(f"{'Samples per class:':<30}" f"{int(sample_value):>15}")
elif wetland_name == "Botswana":
    print(f"{'Samples per class:':<30}" f"{int(sample_value):>15}")
elif wetland_name == "NC16":
    print(f"{'Sampling protocol:':<30}" f"{'Spatially Disjoint':>15}")

print("-" * 70)
print(f"{'Best Validation Epoch:':<30}" f"{best_epoch:>15d}")
print(f"{'Best Validation OA:':<30}" f"{best_val_OA * 100:>14.2f}%")
print("-" * 70)
print(f"{'Test Overall Accuracy (OA):':<30}" f"{OA_test * 100:>14.2f}%")
print(f"{'Test Average Accuracy (AA):':<30}" f"{AA_test * 100:>14.2f}%")
print(f"{'Test Kappa Coefficient:':<30}" f"{Kappa_test:>15.4f}")
print(f"{'Peak GPU Memory:':<30}" f"{peak_gpu_memory:>14.2f} MB")
print(f"{'Preprocessing Time:':<30}" f"{preprocessing_time:>14.2f}s")

print("-" * 70)

if CA_test is not None:
    print(f"{'Class ID':<15} | " f"{'Classification Accuracy (CA)':<20}")

    print("-" * 70)

    for i, acc in enumerate(CA_test):
        print(f"Class {i + 1:<8} | " f"{acc * 100:>10.2f}%")

    print("-" * 70)
print(f"SUCCESS: All results saved in: " f"{time_folder}")
print("=" * 70)

