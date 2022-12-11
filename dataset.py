from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import config
import os

# dataset and dataloader
class LolDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.low_light_files = os.listdir(os.path.join(self.root_dir, "low"))
        self.normal_light_files = os.listdir(os.path.join(self.root_dir, "high"))
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.IMAGE_SIZE),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.low_light_files)

    def __getitem__(self, index):
        low_img_file = self.low_light_files[index]
        low_img_path = os.path.join(self.root_dir, "low", low_img_file)
        low_image_rgb = Image.open(low_img_path)
        low_image_rgb = self.transform(low_image_rgb)
        
        normal_img_file = self.normal_light_files[index]
        normal_img_path = os.path.join(self.root_dir, "high", normal_img_file)
        normal_image_rgb = Image.open(normal_img_path)
        normal_image_gray = normal_image_rgb.convert('L')
        normal_image_rgb = self.transform(normal_image_rgb)
        normal_image_gray = self.transform(normal_image_gray)

        return low_image_rgb, normal_image_rgb, normal_image_gray