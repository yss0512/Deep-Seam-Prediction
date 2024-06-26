import logging
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as transforms

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
    
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def sobleonlyTrans(subdirectories):
    scale = 1
    delta = 0
    pathA = "C:/Users/User/seam-UNet/data/tests/sobelonlyA"
    pathB = "C:/Users/User/seam-UNet/data/tests/sobelonlyB"
    ddepth = cv2.CV_16S
    sub1 = list(subdirectories[0].glob("*.jpg"))
    sub2 = list(subdirectories[1].glob("*.jpg"))
    for sobelsub in sub1:
        imageA = cv2.imread(str(sobelsub),cv2.IMREAD_COLOR)
        imageA =cv2.GaussianBlur(imageA,(3,3),0)
        grayA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
        Agrad_x = cv2.Sobel(grayA,ddepth,1,0,ksize=3,scale=scale,delta=delta,borderType=cv2.BORDER_DEFAULT)
        Agrad_y = cv2.Sobel(grayA,ddepth,0,1,ksize=3,scale=scale,delta=delta,borderType=cv2.BORDER_DEFAULT)
        Aabs_grad_x= cv2.convertScaleAbs(Agrad_x)
        Aabs_grad_y = cv2.convertScaleAbs(Agrad_y)
        gradA= cv2.addWeighted(Aabs_grad_x,0.5,Aabs_grad_y,0.5,0)
        cv2.imwrite(pathA+"/"+str(sobelsub.name),gradA)
        
    
    for sobelsub in sub2:
        imageB = cv2.imread(str(sobelsub),cv2.IMREAD_COLOR)
        imageB =cv2.GaussianBlur(imageB,(3,3),0)
        grayB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)
        Agrad_x = cv2.Sobel(grayB,ddepth,1,0,ksize=3,scale=scale,delta=delta,borderType=cv2.BORDER_DEFAULT)
        Agrad_y = cv2.Sobel(grayB,ddepth,0,1,ksize=3,scale=scale,delta=delta,borderType=cv2.BORDER_DEFAULT)
        Aabs_grad_x= cv2.convertScaleAbs(Agrad_x)
        Aabs_grad_y = cv2.convertScaleAbs(Agrad_y)
        gradB= cv2.addWeighted(Aabs_grad_x,0.5,Aabs_grad_y,0.5,0)
        cv2.imwrite(pathB+"/"+str(sobelsub.name),gradB)
    
    
def sobelTrans(subdirectories):
    scale = 1
    delta = 0
    pathA = "C:/Users/User/seam-UNet/data/tests/sobelA"
    pathB = "C:/Users/User/seam-UNet/data/tests/sobelB"
    ddepth = cv2.CV_16S
    sub1 = list(subdirectories[0].glob("*.jpg"))
    sub2 = list(subdirectories[1].glob("*.jpg"))
    for sobelsub in sub1:
        imageA = cv2.imread(str(sobelsub),cv2.IMREAD_COLOR)
        imageA =cv2.GaussianBlur(imageA,(3,3),0)
        grayA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
        Agrad_x = cv2.Sobel(grayA,ddepth,1,0,ksize=3,scale=scale,delta=delta,borderType=cv2.BORDER_DEFAULT)
        Agrad_y = cv2.Sobel(grayA,ddepth,0,1,ksize=3,scale=scale,delta=delta,borderType=cv2.BORDER_DEFAULT)
        Aabs_grad_x= cv2.convertScaleAbs(Agrad_x)
        Aabs_grad_y = cv2.convertScaleAbs(Agrad_y)
        gradA= cv2.addWeighted(Aabs_grad_x,0.5,Aabs_grad_y,0.5,0)
        cv2.imwrite(pathA+"/"+str(sobelsub.name),cv2.add(gradA,grayA))
        
    
    for sobelsub in sub2:
        imageB = cv2.imread(str(sobelsub),cv2.IMREAD_COLOR)
        imageB =cv2.GaussianBlur(imageB,(3,3),0)
        grayB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)
        Agrad_x = cv2.Sobel(grayB,ddepth,1,0,ksize=3,scale=scale,delta=delta,borderType=cv2.BORDER_DEFAULT)
        Agrad_y = cv2.Sobel(grayB,ddepth,0,1,ksize=3,scale=scale,delta=delta,borderType=cv2.BORDER_DEFAULT)
        Aabs_grad_x= cv2.convertScaleAbs(Agrad_x)
        Aabs_grad_y = cv2.convertScaleAbs(Agrad_y)
        gradB= cv2.addWeighted(Aabs_grad_x,0.5,Aabs_grad_y,0.5,0)
        cv2.imwrite(pathB+"/"+str(sobelsub.name),cv2.add(gradB,grayB))
       

class SobelTransform():
    def __init__(self, trainFolder: str):
        self.trainFolder = Path(trainFolder)
        self.subdirectories = [subdir for subdir in self.trainFolder.glob("*") if subdir.is_dir()]
        # sobelTrans(self.subdirectories)
        sobleonlyTrans(self.subdirectories)
        
    

class SeamDataset(Dataset):
    def __init__(self, sobelimages_dir: str):
        self.sobelimages_dir = Path(sobelimages_dir)
        self.subdirectories = [subdir for subdir in self.sobelimages_dir.glob("*") if subdir.is_dir()] #모든 dir 경로 추출
        
        self.imageA_rgb = [splitext(file)[0] for file in listdir(self.subdirectories[0]) if isfile(join(self.subdirectories[0], file)) and not file.startswith('.')] # imageA
        self.imageB_rgb = [splitext(file)[0] for file in listdir(self.subdirectories[1]) if isfile(join(self.subdirectories[1], file)) and not file.startswith('.')] # imageB
        self.imageA = [splitext(file)[0] for file in listdir(self.subdirectories[4]) if isfile(join(self.subdirectories[4], file)) and not file.startswith('.')] # sobelA 
        self.imageB = [splitext(file)[0] for file in listdir(self.subdirectories[5]) if isfile(join(self.subdirectories[5], file)) and not file.startswith('.')] # sobelB
        self.sobelonlyA = [splitext(file)[0] for file in listdir(self.subdirectories[6]) if isfile(join(self.subdirectories[6], file)) and not file.startswith('.')] # sobelonlyA
        self.sobelonlyB = [splitext(file)[0] for file in listdir(self.subdirectories[7]) if isfile(join(self.subdirectories[7], file)) and not file.startswith('.')] # sobelonlyB
        
        
        
    def __len__(self):
        return len(self.imageA)
    
    @staticmethod
    def preprocess(pil_sobelimgA, pil_sobelimgB,pil_imageA,pil_imageB,pil_maskA,pil_maskB):        
        saw,sah = pil_sobelimgA.size
        sbw,sbh = pil_sobelimgB.size
        iaw,iah = pil_imageA.size
        ibw,ibh = pil_imageB.size
        maw,mah = pil_maskA.size
        mbw,mbh = pil_maskB.size
        scale = 0.5
        
        pil_sobelimgA = pil_sobelimgA.resize((int(saw*scale), int(sah*scale)), resample = Image.BICUBIC)
        pil_sobelimgB = pil_sobelimgB.resize((int(sbw*scale), int(sbh*scale)), resample = Image.BICUBIC)
        pil_imageA = pil_imageA.resize((int(iaw*scale), int(iah*scale)), resample = Image.BICUBIC)
        pil_imageB = pil_imageB.resize((int(ibw*scale), int(ibh*scale)), resample = Image.BICUBIC)
        pil_maskA = pil_maskA.resize((int(maw*scale), int(mah*scale)), resample = Image.BICUBIC)
        pil_maskB = pil_maskB.resize((int(mbw*scale), int(mbh*scale)), resample = Image.BICUBIC)
        sobelimgA = np.asarray(pil_sobelimgA)
        sobelimgB = np.asarray(pil_sobelimgB)
        imageA = np.asarray(pil_imageA)
        imageB = np.asarray(pil_imageB)
        maskA = np.asarray(pil_maskA)
        maskB = np.asarray(pil_maskB)
      
        
        # if sobelimgA.ndim == 2:
        #     sobelimgA = sobelimgA[np.newaxis, ...] # 2->3 높이, 너비 -> 채널, 높이, 너비
        # else:
        #     sobelimgA = sobelimgA.transpose((2, 0, 1)) # basis 높이, 너비, 채널 -> 채널, 높이, 너비  
    
        # if (sobelimgA > 1).any():
        #     sobelimgA = sobelimgA / 255.0
            
            
        # if sobelimgB.ndim == 2:
        #     sobelimgB = sobelimgB[np.newaxis, ...]
        # else:
        #     sobelimgB = sobelimgB.transpose((2, 0, 1))
        
        # if (sobelimgB > 1).any():
        #     sobelimgB = sobelimgB / 255.0
            
        if imageA.ndim == 2:
            imageA = imageA[np.newaxis, ...] # 2->3 높이, 너비 -> 채널, 높이, 너비
        else:
            imageA = imageA.transpose((2, 0, 1)) # basis 높이, 너비, 채널 -> 채널, 높이, 너비  
    
        if (imageA > 1).any():
            imageA = imageA / 255.0
        
        
        if imageB.ndim == 2:
            imageB = imageB[np.newaxis, ...]
        else:
            imageB = imageB.transpose((2, 0, 1))
        
        if (imageB > 1).any():
            imageB = imageB / 255.0
       
        
        # if maskA.ndim == 2:
        #     maskA = maskA[np.newaxis, ...]
        # else:
        #     maskA = maskA.transpose((2, 0, 1))
        
        # if maskB.ndim == 2:
        #     maskB = maskB[np.newaxis, ...]
        # else:
        #     maskB = maskB.transpose((2, 0, 1))
            
        imgSub = np.abs(np.subtract(imageA,imageB))
        imgSub /= 255.0
        
       
           
        return sobelimgA,sobelimgB,imgSub,imageA,imageB,maskA,maskB
    
    def __getitem__(self, idx):
        nameA = self.imageA[idx]
        nameB = self.imageB[idx]
        
        # 해당 이미지 경로 추출
        imgA_file = list(self.subdirectories[0].glob(nameA + '.*'))
        imgB_file = list(self.subdirectories[1].glob(nameB + '.*'))
        maskA_file = list(self.subdirectories[2].glob(nameA + '.*'))
        maskB_file = list(self.subdirectories[3].glob(nameB + '.*'))
        sobelimgA_file = list(self.subdirectories[4].glob(nameA + '.*'))
        sobelimgB_file = list(self.subdirectories[5].glob(nameB + '.*'))
      
        
        # file 존재 유무 검증
        assert len(sobelimgA_file) == 1, f'Either no image or multiple images found for the ID {nameA}: {sobelimgA_file}'
        assert len(sobelimgB_file) == 1, f'Either no image or multiple images found for the ID {nameB}: {sobelimgB_file}'
        assert len(maskA_file) == 1, f'Either no image or multiple images found for the ID {nameA}: {maskA_file}'
        assert len(maskB_file) == 1, f'Either no image or multiple images found for the ID {nameB}: {maskB_file}'
        assert len(imgA_file) == 1, f'Either no image or multiple images found for the ID {nameA}: {imgA_file}'
        assert len(imgB_file) == 1, f'Either no image or multiple images found for the ID {nameB}: {imgB_file}'
        
        # PTL 형식으로 image load
        sobelimgA = load_image(sobelimgA_file[0])
        sobelimgA = sobelimgA.convert('L')
        sobelimgB = load_image(sobelimgB_file[0])
        sobelimgB = sobelimgB.convert('L')
        imgA = load_image(imgA_file[0])
        imgB = load_image(imgB_file[0])
        maskA = load_image(maskA_file[0])
        maskB = load_image(maskB_file[0])
      
        # size 검증
        assert sobelimgA.size == sobelimgB.size,  f'sobelImageA {nameA} and sobelImageB {nameB} should be the same size, but are {sobelimgA.size} and {sobelimgB.size}'
        assert imgA.size == imgB.size,  f'sobelImageA {nameA} and sobelImageB {nameB} should be the same size, but are {imgA.size} and {imgB.size}'
        assert maskA.size == maskB.size,  f'sobelImageA {nameA} and sobelImageB {nameB} should be the same size, but are {maskA.size} and {maskB.size}'
        # preprocess
        sobelimgA, sobelimgB, imgSub, imageA, imageB, maskA, maskB= self.preprocess(sobelimgA, sobelimgB, imgA, imgB, maskA, maskB)
    
        
        return {
            'sobelimageA':torch.as_tensor(sobelimgA.copy()).float().contiguous(),
            'sobelimageB':torch.as_tensor(sobelimgB.copy()).float().contiguous(),
            'imageSub':torch.as_tensor(imgSub.copy()).float().contiguous(),
            'imageA':torch.as_tensor(imageA.copy()).float().contiguous(),
            'imageB':torch.as_tensor(imageB.copy()).float().contiguous(),
            'maskA':torch.as_tensor(maskA.copy()).float().contiguous(),
            'maskB':torch.as_tensor(maskB.copy()).float().contiguous(),
        }
        
        
        
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
