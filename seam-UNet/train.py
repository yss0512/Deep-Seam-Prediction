import argparse
import logging
import os
import random
import sys
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from scipy import ndimage
import wandb
import constant
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, SobelTransform, SeamDataset
from utils.dice_score import dice_loss
import matplotlib.pyplot as plt

#기존 데이터 폴더
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path(constant.SNAPSHOT_DIR)

#UDIS 데이터 폴더
training_folder = Path(constant.TRAIN_FOLDER)
checkpoint_folder = Path(constant.TEST_FOLDER)
test_folder = Path(constant.TEST_FOLDER)


def plt_out(title, image): 
    
    if title=="binaryMaskA" or title =="binrayMaskB":
        image = image.permute(0,2,3,1) # 채널 높이 너비 -> 높이 너비 채널로 변경
        plt.imshow(image[0][:, :, 0].detach().numpy(), cmap="gray")
    else:
        plt.imshow(image.permute(1,2,0),cmap="gray")
    
    plt.title(title)
    plt.show()
    
def patch(overlap):
    P = torch.ones((9, 9), dtype=torch.float32).to(device=device)
    M = 9
    # #cpu
    # overlap = overlap.to('cpu')
    # overlapco = ndimage.convolve(overlap, P[:, :, np.newaxis])
    # overlapco /= (M*M)
    # return torch.tensor(overlapco)
    

    P = P[:, :, np.newaxis]
    P/=255
    
    for i in range(2):
        P  = torch.cat([P, P[:, :, :1]],dim=2)
    
    overlapco = F.conv2d(overlap.unsqueeze(0).permute(0,3,1,2),P.unsqueeze(0).permute(0,3,1,2), padding=4)
    overlapco = overlapco.squeeze(0)
    for i in range(2):
        overlapco = torch.cat([overlapco,overlapco[:1, :, :]],dim=0)

    # print(overlapco)
    # plt.imshow(overlapco.permute(1,2,0).to('cpu'),cmap="gray")
    # plt.show()
   
    
    oversum = torch.sum(overlapco)
    oversum /= (M*M)
    return oversum

def train_model(
        model,
        device,
        epochs: int = constant.ITERATIONS,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = True ,
        amp: bool = False, 
        weight_decay: float = 0, 
        momentum: float = 0.9,
        gradient_clipping: float = 1.0, 
):
    
    # SobelTransform(training_folder)
    
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)
    
    dataset = SeamDataset(training_folder)
   
    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    n_val = 0
    n_train = len(dataset)
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set = dataset
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # first = next(iter(train_loader))
    # print(first['imageA'].numpy())
    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )
    experiment = wandb.init(project='Seam_U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, amp=amp)
    )


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,foreach=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    start = time.time() 
    all_one = torch.ones((112,112),device=device)
    end = time.time()
   

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        # with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
            
            
            # images, true_masks = batch['image'], batch['mask']
            sobelimageA, sobelimageB, imageSub, imageA, imageB, maskA, maskB = \
                batch['sobelimageA'],batch['sobelimageB'],batch['imageSub'], batch['imageA'], batch['imageB'],batch['maskA'],batch['maskB']
            

            assert imageSub.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded images have {imageSub.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'
            
            start = time.time()   
            imageSub = imageSub.to(device=device, dtype=torch.float32,memory_format=torch.channels_last)
            imageA = imageA.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            imageB = imageB.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            maskA = maskA.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            maskB = maskB.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # imageSub /=255.0
            maskA /=255.0
            maskB /=255.0
            end = time.time()
           
          
            

            #true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):  
                    
                # start = time.time()
                # print("imageSub device: ", imageSub.device)
            
                masks_pred = model(imageSub)
                
                # plt.imshow(masks_pred.to('cpu').permute(0,2,3,1)[0][:, :, 0].detach().numpy(),cmap='gray')
                # plt.show()
                
                
                
                
                # end = time.time()
                # print(f"unet time: {end - start:.5f} sec")
                
                # M*A + M*B = all one_matrix
                # binaryMaskA = masks_pred
                binaryMaskA = torch.sigmoid(masks_pred)
                binaryMaskA = (binaryMaskA > 0.5).float() # binary change -> M*A
            
                # start = time.time() 
                all_one = torch.ones(binaryMaskA.shape,device=device)
                # end = time.time()
                # print(f"all-one time: {end - start:.5f} sec")

                binaryMaskB = torch.subtract(all_one, binaryMaskA).float()
            
                # start = time.time()    
                # 4 dim -> 3 dim 및 순서 변경
                with torch.no_grad():
                    m_ac = maskA[0].permute(2,0,1)
                    m_bc = maskB[0].permute(2,0,1)
             
                
                # same binrayMaskA[0][0]
                binaryMaskA = torch.tensor(binaryMaskA.permute(0, 2, 3, 1)[0][:, :, 0].detach())
                binaryMaskB = torch.tensor(binaryMaskB.permute(0, 2, 3, 1)[0][:, :, 0].detach())
                binaryMaskA = binaryMaskA.unsqueeze(0)
                binaryMaskB = binaryMaskB.unsqueeze(0)
                
                # 1 channel -> 3 channel
                for i in range(2):
                    binaryMaskA = torch.cat([binaryMaskA, binaryMaskA[:1, :, :]], dim=0)
                    binaryMaskB = torch.cat([binaryMaskB, binaryMaskB[:1, :, :]], dim=0)
                    
                assert m_ac.shape == binaryMaskA.shape, f'size:{m_ac.shape}, binarysize:{binaryMaskA.shape}'
                
                
                # end = time.time()
                # print(f"마스크 time: {end - start:.5f} sec")
                # 3 dim * 3 dim
                
                # start = time.time()
                ma = torch.mul(binaryMaskA,m_ac)
                mb = torch.mul(binaryMaskB,m_bc)
                

            
                # end = time.time()
                # print(f"ma,mb time: {end - start:.5f} sec")
                    
                # imageC = torch.add(imageA[0],imageB[0])
                # # start = time.time()  
                # Orignal_nonoverlapA = torch.where((imageC > 0) & (imageA[0] == 0), imageC, torch.tensor(0, dtype=torch.float32)) # orginal R11
                # Orignal_nonoverlapB = torch.where((imageC > 0) & (imageB[0] == 0), imageC, torch.tensor(0, dtype=torch.float32)) # orginal R22\
                
                # # end = time.time()
                # # print(f"orginal_nonover time: {end - start:.5f} sec")
                
                # originalC = imageC-Orignal_nonoverlapA - Orignal_nonoverlapB
                # originalC = originalC/2

        
               
                # start = time.time()
                imageA = imageA.permute(0,2,3,1)
                imageB = imageB.permute(0,2,3,1)
              
                ia = torch.mul(imageA[0],ma.permute(1,2,0))
                ib = torch.mul(imageB[0],mb.permute(1,2,0))
                ic = torch.add(ia,ib)
                
                nonoverlapA = torch.where((ic > 0) & (imageA[0] == 0), ic, torch.tensor(0, dtype=torch.float32)) # R11
                nonoverlapB = torch.where((ic > 0) & (imageB[0] == 0), ic, torch.tensor(0, dtype=torch.float32)) # R22
               
                
                # total pixelNumbers in R11 and R12
                R11_PixelNumbers = (nonoverlapA>0).sum().item()
                R22_PixelNumbers = (nonoverlapB>0).sum().item()
               
                

                overlapA = torch.where((ic > 0) & (imageA[0] > 0), ic, torch.tensor(0, dtype=torch.float32)) - nonoverlapB # A기준으로 C R12
                overlapB = torch.where((ic > 0) & (imageB[0] > 0), ic, torch.tensor(0, dtype=torch.float32)) - nonoverlapA # B기준으로 C R12 
                overlapC =  torch.where((ic > 0) & (overlapA > 0), ic, torch.tensor(0, dtype=torch.float32)) # A기준 C R12
              
                
                # (x,x,3)
                overlapiA = torch.where((imageA[0] > 0) & (overlapC > 0), imageA[0], torch.tensor(0, dtype=torch.float32)) # imageA R12
                overlapiB = torch.where((imageB[0] > 0) & (overlapC > 0), imageB[0], torch.tensor(0, dtype=torch.float32)) # imageB R12
                
               
                # total pixelnumbers in R12
                R12_PixelNumbers = (overlapA>0).sum().item()
            
                # end = time.time()
                # print(f"overlap time: {end - start:.5f} sec")
               
                if R11_PixelNumbers !=0 and R22_PixelNumbers !=0:
                    # non_overlapping area loss
                    loss_non = ((torch.sum(torch.abs(ic[nonoverlapA > 0] - imageB[0][nonoverlapA > 0])).item()) / R11_PixelNumbers) + \
                                + ((torch.sum(torch.abs(ic[nonoverlapB > 0] - imageA[0][nonoverlapB > 0])).item())/ R22_PixelNumbers)
                else:
                    loss_non = 0.0
                    
                loss_non = torch.tensor(loss_non)
                loss_non.requires_grad = True
                
    
                # overlapping area loss_pixel
                loss_pixel = min(torch.sum(torch.abs(ic[overlapC > 0] - imageA[0][overlapC > 0])).item(), \
                            torch.sum(torch.abs(ic[overlapC > 0] - imageB[0][overlapC > 0])).item()) / R12_PixelNumbers
                
                
                # start = time.time()
                
                ic_P = patch(overlapC)
                ia_P = patch(overlapiA)
                ib_P = patch(overlapiB)
             
                
                # end = time.time()
                # print(f" patch time: {end - start:.5f} sec")
            
                
                # overlapping area loss_patch
                loss_patch = min(torch.sum(torch.abs(ic_P - ia_P)).item(),torch.sum(torch.abs(ic_P - ib_P)).item()) / R12_PixelNumbers
                loss_patch = torch.tensor(loss_patch)
                loss_patch.requires_grad = True

                #final loss function
                w1 = 200
                w2 = 100
                loss = w1 * loss_non + w2 * loss_patch    
                
                # if model.n_classes == 1:
                #     loss = criterion(masks_pred.squeeze(1), true_masks.float())
                #     loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                # else:
                #     loss = criterion(masks_pred, true_masks)
                #     loss += dice_loss(
                #         F.softmax(masks_pred, dim=1).float(),
                #         F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                #         multiclass=True
                #     )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            experiment.log({
                'train loss': loss.item(),
                'step': global_step,
                'epoch': epoch
            })
            print({
                'train loss': loss.item(),
                'step': global_step,
                'epoch': epoch
            })
            # pbar.set_postfix(**{'loss (batch)': loss.item()})
            # val_score = evaluate(model, val_loader, device, amp)
            # try:
            #     experiment.log({
            #         'learning rate': optimizer.param_groups[0]['lr'],
            #         'validation Dice': val_score,
            #         'images': wandb.Image(overlapC.cpu()),
            #         'step': global_step,
            #         'epoch': epoch})
            # except:
            #     pass
            

            # # Evaluation round
            # division_step = (n_train // (5 * batch_size))
            # if division_step > 0:
            #     if global_step % division_step == 0:
            #         histograms = {}
            #         for tag, value in model.named_parameters():
            #             tag = tag.replace('/', '.')
            #             if not (torch.isinf(value) | torch.isnan(value)).any():
            #                 histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            #             if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
            #                 histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            #         val_score = evaluate(model, val_loader, device, amp)
            #         # scheduler.step(val_score)

            #         logging.info('Validation Dice score: {}'.format(val_score))
            #         try:
            #             experiment.log({
            #                 'learning rate': optimizer.param_groups[0]['lr'],
            #                 'validation Dice': val_score,
            #                 'images': wandb.Image(overlapC.cpu()),
            #                 'step': global_step,
            #                 'epoch': epoch,
            #                 **histograms
            #             })
            #         except:
            #             pass
            # check gpu memory
            time.sleep(0.01)
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=constant.ITERATIONS, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    print(device)
    torch.cuda.empty_cache()
    
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        # del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.CudaError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            amp=args.amp
        )
