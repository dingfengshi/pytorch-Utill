from torch.utils.data import Dataset
from torchvision.transforms import transforms


# 定义数据集
class mydataset(Dataset):
    def __init__(self, frame_file, path, transform=None):
        self.frame_file = frame_file  # 存储数据集总体信息的文件
        self.path = path  # 数据集位置
        # transform
        self.trans = transforms

    def __len__(self):
        # 实现dataset长度
        return len(self.frame_file)

    def __getitem__(self, idx):
        # 实现数据提取和转换
        img = self.frame_file[idx]
        self.trans(img)
        pass
