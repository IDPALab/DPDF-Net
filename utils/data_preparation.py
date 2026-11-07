import numpy as np
import torch
from scipy.io import loadmat

class HSI_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, train, patch_size):

        super(HSI_Dataset, self).__init__()
        self.data = data
        self.label = label
        self.patch_size = patch_size

        mask = np.ones_like(label)
        if train:
            mask[label == 0] = 0

        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]

        if train:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        if self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]

        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        if self.patch_size > 1:
            data = data.unsqueeze(0)
        
        return data, label
