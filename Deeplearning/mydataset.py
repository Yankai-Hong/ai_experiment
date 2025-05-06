import os
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]

        # 提取所有文件名前缀（如 'baihe01.jpg' → 'baihe'）作为类别
        self.labels = [os.path.basename(p).split('0')[0] for p in self.img_paths]

        # 创建类别到整数的映射
        unique_labels = sorted(set(self.labels))
        self.label_to_index = {name: idx for idx, name in enumerate(unique_labels)}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label_name = os.path.basename(img_path).split('0')[0]
        label = self.label_to_index[label_name]

        return image, label
