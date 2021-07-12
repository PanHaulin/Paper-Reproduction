from PIL import Image
from random import normalvariate, randint
import os
import numpy as np
import sys
from torchvision.transforms import TenCrop, Normalize
import glob
sys.path.append('.')

class ShortEdgeScale(object):
    """以短边为准等比例缩放
    Args:
        size_min （int）: 最小值
        size_max (int): 最大值
    """
 
    def __init__(self, size_min, size_max):
        super().__init__()
        self.size_min = size_min
        self.size_max = size_max
 
    def __call__(self, img:Image):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        w = img.size[0]
        h = img.size[1]
        random_size = randint(self.size_min, self.size_max)
        if w < h:
            return img.resize((random_size, round(h/w * random_size)))
        else:
            return img.resize((round(w/h * random_size), random_size))

class MeanSubstractAndColorShift(object):
    """求通道均值并做PCA进行图像增广
    Args:
        DATASET_PATH
    """

    def __init__(self, DATASET_PATH) -> None:
        super().__init__()
        self.DATASET_PATH = DATASET_PATH
    
    def get_mean(self):
        paths = glob.glob(self.DATASET_PATH + '*/*')
        for path in paths:
            file_name = path.split('/')[-1]
            folder = path.split('/')[-2]
            if file_name.endswith("JPEG"):
                file = os.path.join(self.DATASET_PATH, folder, file_name)
                img = Image.open(file)
                img = img.convert("RGB")
                img = np.array(img)
                img = img.reshape((-1,3))
                try:
                    img_array = np.concatenate((img_array, img),0) # 第一张不存在img_array
                except:
                    img_array = img
        mean_value = img_array.mean(0)
        return mean_value
    
    def pca(self, mean_value):
        paths = glob.glob(self.DATASET_PATH + '*/*')
        for path in paths:
            file_name = path.split('/')[-1]
            folder = path.split('/')[-2]
            if file_name.endswith("JPEG"):
                file = os.path.join(self.DATASET_PATH, folder, file_name)
                img = Image.open(file)
                img = img.convert("RGB")
                img = np.array(img)
                img = img.reshape((-1,3))
                img = img.astype(float)
                img -= mean_value
                try:
                    img_array = np.concatenate((img_array, img),0) # 第一张不存在img_array
                except:
                    img_array = img
        
        # 求协方差矩阵
        img_cov = np.cov([img_array[:,0],img_array[:,1],img_array[:2]])
        lambd, p = np.linalg.eig(img_cov)
        alpha0 = normalvariate(0,0.1)
        alpha1 = normalvariate(0,0.1)
        alpha2 = normalvariate(0,0.1)
        v = np.transpose((alpha0*lambd[0], alpha1*lambd[1], alpha2*lambd[2]))
        add_num = np.dot(v, np.transpose(p))
        return add_num


    def __call__(self, img):
        if not os.path.exists('temp/mean_add.npy'):
            mean_value = self.get_mean()
            add_num = self.pca()
            temp = np.array([mean_value, add_num])
            np.save('temp/mean_add.npy', temp)
        else:
            temp = np.load('temp/mean_add.npy')
            mean_value = temp[0]
            add_num = temp[1]
        
        img = np.array(img)
        for i in range(3):
            add_num = add_num.astype(int)
            img[:,:,i] = (img[:,:,i] - mean_value[i] + add_num[i])
        img = Image.fromarray(np.uint8(img))
        return img

class MultiScaleCrops(object):
    """以短边为准等比例缩放
    Args:
        size_list: 缩放列表
    """
 
    def __init__(self, size_list):
        super().__init__()
        self.size_list = size_list
 
    def __call__(self, img:Image):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            Tuple of PIL Image: Tuple
        """
        w = img.size[0]
        h = img.size[1]
        for random_size in self.size_list:
            random_size = self.size_list[randint(0, len(self.size_list)-1)]
            if w < h:
                img = img.resize((random_size, round(h/w * random_size)))
            else:
                img = img.resize((round(w/h * random_size), random_size))
            # print("{},{}".format(w,h))
            crop = TenCrop(size=self.size_list[0])(img)
            
            try:
                all_crop = (*all_crop, *crop) # 第一个size不存在all_crop
            except:
                all_crop = crop
        return all_crop
            