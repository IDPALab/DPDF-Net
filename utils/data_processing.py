import numpy as np
from scipy.ndimage import binary_dilation

# the position of train and test samples
#-------------------------------------------------------------------------------


def position_train_and_test_point(train_data, test_data, label_data, num_classes):
    number_train = []
    position_train = {}
    number_test = []
    position_test = {}
    number_true = []
    position_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        position_train[i] = each_class

    total_position_train = position_train[0]
    for i in range(1, num_classes):
        total_position_train = np.r_[total_position_train, position_train[i]]

    total_position_train = total_position_train.astype(int)

    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        position_test[i] = each_class

    total_position_test = position_test[0]
    for i in range(1, num_classes):
        total_position_test = np.r_[total_position_test, position_test[i]]
    total_position_test = total_position_test.astype(int)

    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(label_data==i)
        number_true.append(each_class.shape[0])
        position_true[i] = each_class

    total_position_true = position_true[0]
    for i in range(1, num_classes+1):
        total_position_true = np.r_[total_position_true, position_true[i]]
    total_position_true = total_position_true.astype(int)

    return total_position_train, total_position_test, total_position_true, number_train, number_test, number_true


def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch//2
    mirror_hsi = np.zeros((height+2*padding,width+2*padding,band),dtype=np.float16)

    mirror_hsi[padding:(padding+height),padding:(padding+width),:] = input_normalize

    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:] = input_normalize[:,padding-i-1,:]

    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:] = input_normalize[:,width-1-i,:]

    for i in range(padding):
        mirror_hsi[i,:,:] = mirror_hsi[padding*2-i-1,:,:]

    for i in range(padding):
        mirror_hsi[height+padding+i,:,:] = mirror_hsi[height+padding-1-i,:,:]

    print("===============================================================================")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("===============================================================================")
    return mirror_hsi


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):

    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2

    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=np.float16)
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape

    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]

    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band


def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=np.float16)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=np.float16)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=np.float16)

    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("===============================================================================")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("===============================================================================")
    return x_train_band, x_test_band, x_true_band


def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []

    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)

    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("===============================================================================")
    return y_train, y_test, y_true


def calculate_patch_overlap_ratio(train_mask, test_mask, patch_size=7):
    radius = patch_size // 2

    structure = np.ones((patch_size, patch_size), dtype=bool)
    train_patch_footprint = binary_dilation(train_mask > 0, structure=structure)
    test_points = np.argwhere(test_mask > 0)
    total_test_patches = len(test_points)

    if total_test_patches == 0:
        return 0.0

    test_patch_centers_hit = train_patch_footprint[
        test_points[:, 0], test_points[:, 1]
    ]
    overlap_count = np.sum(test_patch_centers_hit)

    overlap_ratio = (overlap_count / total_test_patches) * 100.0
    print('=================================================================')
    print(f'[Spatial Overlap Analysis (Patch Size = {patch_size}x{patch_size})]')
    print(f'Total test patches: {total_test_patches}')
    print(f'Test patch centers within the training footprint: {overlap_count}')
    print(f'Spatial overlap ratio: {overlap_ratio:.2f}%')
    print('=================================================================')
    return overlap_ratio



# Split NC16 into separate regions with a buffer between them.
def create_nc16_spatially_disjoint_split(
    gt_label,
    split_direction='horizontal',
    split_ratio=0.5,
    buffer_width=20,
    val_ratio=0.2,
    seed=42,
):
    np.random.seed(seed)
    H, W = gt_label.shape
    num_classes = int(np.max(gt_label))

    train_label = np.zeros_like(gt_label)
    val_label = np.zeros_like(gt_label)
    test_label = np.zeros_like(gt_label)

    if split_direction == 'vertical':
        split_point = int(W * split_ratio)
        bound_train = split_point - buffer_width // 2
        bound_test = split_point + buffer_width // 2

        region_A = np.zeros((H, W), dtype=bool)
        region_B = np.zeros((H, W), dtype=bool)
        region_A[:, :bound_train] = True
        region_B[:, bound_test:] = True
    else:
        split_point = int(H * split_ratio)
        bound_train = split_point - buffer_width // 2
        bound_test = split_point + buffer_width // 2

        region_A = np.zeros((H, W), dtype=bool)
        region_B = np.zeros((H, W), dtype=bool)
        region_A[:bound_train, :] = True
        region_B[bound_test:, :] = True

    print('\n=== [NC16 Spatially Disjoint Split Summary] ===')
    print(
        f"{'Class':<8} | {'Total':<8} | {'Train (80% A)':<14} | {'Val (20% A)':<12} | {'Test (Region B)':<15}"
    )
    print('-' * 65)

    for c in range(1, num_classes + 1):
        pos_A = np.argwhere((gt_label == c) & region_A)
        pos_B = np.argwhere((gt_label == c) & region_B)

        nA = len(pos_A)
        nB = len(pos_B)

        if nA == 0 or nB == 0:
            print(
                f'Warning: Class {c} has {nA} samples in region A and {nB} in region B.'
                'One region has no samples for this class. Change split_direction.'
            )

        # Split region A into training and validation samples.
        indices = np.arange(nA)
        np.random.shuffle(indices)
        n_val = int(nA * val_ratio)
        n_train = nA - n_val

        for idx in indices[:n_train]:
            train_label[pos_A[idx][0], pos_A[idx][1]] = c

        for idx in indices[n_train:]:
            val_label[pos_A[idx][0], pos_A[idx][1]] = c

        # Use all labeled samples in region B for testing.
        for pt in pos_B:
            test_label[pt[0], pt[1]] = c

        print(
            f'Class {c:<3} | {nA+nB:<8} | {n_train:<14} | {n_val:<12} | {nB:<15}'
        )

    print('-' * 65)

    print('\nChecking for spatial overlap:')
    calculate_patch_overlap_ratio(train_label + val_label, test_label, patch_size=7)

    return train_label, val_label, test_label
