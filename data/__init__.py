import copy
import importlib
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import center_crop
from torchvision.utils import make_grid



class CustomDatasetLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""  
    
    def __init__(self, dir, csv_file, target_size=None, transforms=None):
        self.img_names = pd.read_csv(csv_file)
        self.transforms = transforms
        self.target_size = target_size
        self.dir=dir

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.img_names)

    def __getitem__(self, idx):
        """Return a batch of data"""
        rimg_path = os.path.join(self.dir,self.img_names.iloc[idx,0])
        cimg_path = os.path.join(self.dir,self.img_names.iloc[idx,1])
        real = Image.open(rimg_path).convert('RGB')
        condition = Image.open(cimg_path).convert('RGB')
        T1 = transforms.Resize((128,128))
        real = T1(real)
        

        target_size = self.target_size
        if target_size:
          T2 = transforms.Resize((256,256))
          real = T2(real)
        if transforms:
          real = self.transforms(real)
          condition = self.transforms(condition)

        return real, condition            