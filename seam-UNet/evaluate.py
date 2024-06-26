import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

def patch(overlap):
    P = np.ones((9, 9),dtype=np.int32)
    M = 9
    overlap = overlap * 255
    overlap = overlap.to(torch.int32)
    overlap = overlap.numpy()
    overlapcopy = np.zeros_like(overlap)
    
    # Perform convolution
    for channel in range(overlap.shape[0]):
        overlapcopy[channel, :, :] = ndimage.convolve(overlap[channel, :, :], P)
        overlapcopy[channel, :,  ] //= (M*M)
        
    overlapcopy  = overlapcopy / 255
    overlapcopy = torch.as_tensor(overlapcopy)
    
    return overlapcopy
@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            sobelimageA, sobelimageB, imageSub, imageA, imageB, maskA, maskB = \
                    batch['sobelimageA'],batch['sobelimageB'],batch['imageSub'], batch['imageA'], batch['imageB'],batch['maskA'],batch['maskB']

            # move images and labels to correct device and type
            imageSub = imageSub.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            # predict the mask
            mask_pred = net(imageSub)
            binaryMaskA = torch.sigmoid(mask_pred)
            binaryMaskA = (binaryMaskA > 0.5).int() # binary change -> M*A
            
            binaryMaskB = torch.subtract(torch.ones(binaryMaskA.shape),binaryMaskA).int() # -> M*B
        
            
        
            transform = transforms.ToTensor()
            # 4 dim -> 3 dim으로 저장 
            m_ac = maskA[0] 
            m_bc = maskB[0]
            # rgb channel 1 channel로 변경(3,A,B)->(1,A,B)
            m_ac = m_ac[0].unsqueeze(0)
            m_bc = m_bc[0].unsqueeze(0)
            # numpy (A,B)->tensor(1,A,B)
            binaryMaskA = transform(binaryMaskA.permute(0,2,3,1)[0][:, :, 0].detach().numpy())
            binaryMaskB = transform(binaryMaskB.permute(0,2,3,1)[0][:, :, 0].detach().numpy())
            
            assert m_ac.shape == binaryMaskA.shape, f'size:{m_ac.shape}, binarysize:{binaryMaskA.shape}'


            # 3 dim * 3 dim
            ma = torch.mul(binaryMaskA,m_ac)
            mb = torch.mul(binaryMaskB,m_bc)
            
            # channel 1을 channel 3으로 변경
            for i in range(2):
                ma = torch.cat([ma, ma[:1, :, :]], dim=0)
                mb = torch.cat([mb, mb[:1, :, :]], dim=0)
                
            imageC = torch.add(imageA[0],imageB[0])
            Orignal_nonoverlapA = torch.where((imageC > 0) & (imageA[0] == 0), imageC, torch.tensor(0)) # orginal R11
            Orignal_nonoverlapB = torch.where((imageC > 0) & (imageB[0] == 0), imageC, torch.tensor(0)) # orginal R22
            originalC = imageC-Orignal_nonoverlapA - Orignal_nonoverlapB
            originalC= originalC/2
            
            # ic = ia*ma + ib*mb
            ia = torch.mul(imageA[0],ma)
            ib = torch.mul(imageB[0],mb)
            ic = torch.add(ia,ib)
            nonoverlapA = torch.where((ic > 0) & (imageA[0] == 0), ic, torch.tensor(0)) # R11
            nonoverlapB = torch.where((ic > 0) & (imageB[0] == 0), ic, torch.tensor(0)) # R22
            
            # total pixelNumbers in R11 and R12
            R11_PixelNumbers = (nonoverlapA > 0).sum().item()
            R22_PixelNumbers = (nonoverlapB > 0).sum().item()
            
            
            overlapA = torch.where((ic > 0) & (imageA[0] > 0), ic, torch.tensor(0)) - nonoverlapB # A기준으로 C R12
            overlapB = torch.where((ic > 0) & (imageB[0] > 0), ic, torch.tensor(0)) - nonoverlapA # B기준으로 C R12

            # R12 individual  
            overlapC =  torch.where((ic > 0) & (overlapA > 0), ic, torch.tensor(0)) # A기준 C R12
            
            overlapiA = torch.where((imageA[0] > 0) & (overlapC > 0), imageA[0], torch.tensor(0)) # imageA R12
            overlapiB = torch.where((imageB[0] > 0) & (overlapC > 0), imageB[0], torch.tensor(0)) # imageB R12
            
            # total pixelnumbers in R12
            R12_PixelNumbers = (overlapA > 0).sum().item()
            # non_overlapping area loss
            loss_non = (torch.sum(torch.abs(ic[nonoverlapA > 0] - imageA[0][nonoverlapA > 0])).item() / R11_PixelNumbers) + \
                        + (torch.sum(torch.abs(ic[nonoverlapB > 0] - imageB[0][nonoverlapB > 0])).item() / R22_PixelNumbers)
            loss_non =torch.tensor(loss_non)
            loss_non.requires_grad = True

            # overlapping area loss_pixel
            loss_pixel = min(torch.sum(torch.abs(ic[overlapC > 0] - imageA[0][overlapC > 0])).item(), \
                        torch.sum(torch.abs(ic[overlapC > 0] - imageB[0][overlapC > 0])).item()) / R12_PixelNumbers
            ic_P = patch(overlapC)
            ia_P = patch(overlapiA)
            ib_P = patch(overlapiB)
            
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

    net.train()
    return loss / max(num_val_batches, 1)
