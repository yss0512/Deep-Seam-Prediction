import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import constant
from utils.data_loading import BasicDataset, SeamDataset
from unet import UNet
from pathlib import Path
from utils.utils import plot_img_and_mask
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
test_folder = Path(constant.TEST_FOLDER)
def predict_img(net,
                # full_img,
                device,
                # scale_factor=1,
                out_threshold=0.5):
    
    transform = transforms.ToTensor()
    dataset = SeamDataset(test_folder)
    test_set = dataset
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=True, **loader_args) 
    net.eval()
    batch = dataset[90]
    sobelimageA, sobelimageB, imageSub, imageA, imageB, maskA, maskB = \
            batch['sobelimageA'],batch['sobelimageB'],batch['imageSub'], batch['imageA'], batch['imageB'],batch['maskA'],batch['maskB']
    
    imageSub = imageSub.to(device=device, dtype=torch.float32)
    imageA = imageA.to(device=device, dtype=torch.float32)
    imageB = imageB.to(device=device, dtype=torch.float32)
    maskA = maskA.to(device=device, dtype=torch.float32)
    maskB = maskB.to(device=device, dtype=torch.float32)
    
    # imageSub /=255.0
    imageA /=255.0
    imageB /=255.0
    imageSub = imageSub.unsqueeze(0)
    maskA /=255.0
    maskB /=255.0
    output,maskA,maskB,imageA,imageB = imageSub,maskA,maskB,imageA,imageB
    mask_predic = net(output)
    mask_predic = mask_predic.squeeze(0)
    for i in range(1):
        mask_predic = torch.cat([mask_predic, mask_predic[:1, :, :]], dim=0)
        
    binaryMaskA = torch.sigmoid(mask_predic)
    binaryMaskA = (binaryMaskA > 0.5).int() # binary change -> M*A
    binaryMaskB = torch.subtract(torch.ones(binaryMaskA.shape,device=device),binaryMaskA).int() # -> M*B
    
    # 4 dim -> 3 dim으로 저장 
    with torch.no_grad():
        m_ac = maskA.permute(2,0,1)
        m_bc = maskB.permute(2,0,1)
    
    
    
    
    assert m_ac.shape == binaryMaskA.shape, f'size:{m_ac.shape}, binarysize:{binaryMaskA.shape}'


    # 3 dim * 3 dim
    ma = torch.mul(binaryMaskA,m_ac)
    mb = torch.mul(binaryMaskB,m_bc)
   
    
    ma = ma.permute(1,2,0)
    mb = mb.permute(1,2,0)
    # ic = ia*ma + ib*mb
    imageA = imageA.permute(1,2,0)
    imageB = imageB.permute(1,2,0)
    print(imageA.shape)
    print(ma.shape)
    ia = torch.mul(imageA,ma)
    ib = torch.mul(imageB,mb)
    ic = torch.add(ia,ib)
    
    nonoverlapB = torch.where((ic > 0) & (imageB == 0), ic, torch.tensor(0)) # R22
    overlapA = torch.where((ic > 0) & (imageA > 0), ic, torch.tensor(0)) - nonoverlapB # A기준으로 IC R12
    overlapC =  torch.where((ic > 0) & (overlapA > 0), ic, torch.tensor(0)) # IC R12
    imageC = torch.add(imageA,imageB)
    Orignal_nonoverlapA = torch.where((imageC > 0) & (imageA == 0), imageC, torch.tensor(0)) # orginal R11
    Orignal_nonoverlapB = torch.where((imageC > 0) & (imageB == 0), imageC, torch.tensor(0)) # orginal R22
    originalC = imageC-Orignal_nonoverlapA - Orignal_nonoverlapB
    originalC= originalC/2
    print(binaryMaskA)
    plt.subplot(1,4,1)
    plt.imshow(binaryMaskA[0].unsqueeze(0).permute(1,2,0).to('cpu'),cmap='gray')
    plt.subplot(1,4,2)
    plt.imshow(binaryMaskB[0].unsqueeze(0).permute(1,2,0).to('cpu'),cmap='gray')
    plt.subplot(1,4,3)
    plt.imshow(originalC.to('cpu'))
    plt.subplot(1,4,4)
    plt.imshow(overlapC.to('cpu'))
    plt.show()
    
    return ma,mb
    

    # with torch.no_grad():
    #     output = net(img).cpu()
    #     output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
    #     if net.n_classes > 1:
    #         mask = output.argmax(dim=1)
    #     else:
    #         mask = torch.sigmoid(output) > out_threshold

    # return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    # parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    # parser.add_argument('--viz', '-v', action='store_true',
    #                     help='Visualize the images as they are processed')
    # parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    # parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
    #                     help='Minimum probability value to consider a mask pixel white')
    # parser.add_argument('--scale', '-s', type=float, default=0.5,
    #                     help='Scale factor for the input images')
    # parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # in_files = args.input
    # out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=2, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    
    
    ma,mb = predict_img(net=net,
                        #    full_img=img,
                        #    scale_factor=args.scale,
                        #    out_threshold=args.mask_threshold,
                           device=device)
    ma = ma[:, :, 0].detach()
    mb = mb[:, :, 0].detach()
    print(ma.shape)
    plt.subplot(1,2,1)
    plt.imshow(ma.to('cpu'),cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mb.to('cpu'),cmap='gray')
    plt.show()
    
    # for i, filename in enumerate(in_files):
    #     logging.info(f'Predicting image {filename} ...')
    #     img = Image.open(filename)

    #     mask_starA = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)
        
        

        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask, mask_values)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')

        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
