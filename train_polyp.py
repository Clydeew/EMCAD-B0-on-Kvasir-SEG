import os
import time
import logging
import argparse
import hashlib
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project-specific imports
from lib.networks import EMCADNet
from utils.dataloader_polyp import get_loader as get_loader
from utils.utils import clip_gradient, AvgMeter


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def structure_loss(pred, mask, w=1):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (w * (wbce + wiou)).mean()

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2. * intersection + smooth) / (total + smooth)

def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def test(model, path, dataset, opt):
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    model.eval()
    
    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root, 
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
        shuffle=False, split='test', color_image=opt.color_image
    )
    
    DSC, IOU, total_images = 0.0, 0.0, 0
    with torch.no_grad():
        for pack in test_loader:
            images, gts, original_shapes, _ = pack       
            images = images.cuda()
            gts = gts.cuda().float()

            ress = model(images)
            if not isinstance(ress, list):
                ress = [ress]
            # Take the primary output
            predictions = ress[-1]
            
            for i in range(len(images)):
                # Note: original_shapes in some loaders is [W, H], in others [H, W]
                # We ensure it matches your specific data loader's return order
                w_orig, h_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
                # 1. Prediction Resize (Bilinear for soft maps)
                p = predictions[i].unsqueeze(0)
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred_resized = pred_resized.sigmoid().squeeze()
                
                # 2. Local Normalization
                pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
                
                # 3. GT Resize (NEAREST to maintain binary mask integrity)
                g = gts[i].unsqueeze(0)
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze()

                #print(pred_resized.shape, gt_resized.shape, g.shape)

                # 4. Binary Thresholding
                input_binary = (pred_resized >= 0.5).float()
                target_binary = (gt_resized >= 0.2).float() 

                # Applying original thresholding (0.5 for pred, 0.2 for target) 
                DSC += dice_coefficient(input_binary, target_binary).item()
                IOU += iou(input_binary, target_binary).item()
                total_images += 1

    return DSC / total_images, IOU / total_images, total_images

def train(train_loader, model, optimizer, epoch, opt, model_name):
    model.train()
    global best, total_train_time
    
    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates = [0.75, 1, 1.25] 
    total_step = len(train_loader)

    for i, (images, gts) in enumerate(train_loader, start=1):
        for rate in size_rates:            
            optimizer.zero_grad()
            images, gts = Variable(images).cuda(), Variable(gts).float().cuda()
    
            if rate != 1:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='nearest')
            
            P = model(images)
            if not isinstance(P, list):
                P = [P]
            loss_p1 = structure_loss(P[0], gts)
            loss_p2 = structure_loss(P[1], gts)
            loss_p3 = structure_loss(P[2], gts)
            loss_p4 = structure_loss(P[3], gts)
            loss_p1234 = structure_loss(P[0]+P[1]+P[2]+P[3], gts)

            weights = [1, 1, 1, 1, 1]
            loss = weights[0]*loss_p1 + weights[1]*loss_p2 + weights[2]*loss_p3 + weights[3]*loss_p4 + weights[4]*loss_p1234

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        if i % 100 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss_record.show():.4f}')
        
    total_train_time += (time.time() - epoch_start)
    
    # Save Last
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-last.pth"))

    # Select checkpoints on validation data only. Evaluate the test split once at the end.
    d_dice, d_iou, _ = test(model, opt.test_path, 'val', opt)
    epoch_results = {'val/dice': d_dice, 'val/iou': d_iou}
    logging.info(f'Epoch: {epoch}, Dataset: val, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
    print(f'Epoch: {epoch}, Dataset: val, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')

    # Check if Best Validation Dice
    if epoch_results['val/dice'] > best:
        logging.info(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val/dice']:.4f}) ###")
        print(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val/dice']:.4f}) ###")
        best = epoch_results['val/dice']
        torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth"))
    return {'train/loss': float(loss_record.show()),
            'train/lr': optimizer.param_groups[0]['lr'],
            'best/val_dice': best, **epoch_results}
    
if __name__ == '__main__':
    # Initial defaults
    dataset_name = 'Kvasir'
    
    parser = argparse.ArgumentParser()
    # network related parameters
    parser.add_argument('--encoder', type=str,
                        default='pvt_v2_b0', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
    parser.add_argument('--expansion_factor', type=int,
                        default=2, help='expansion factor in MSCB block')
    parser.add_argument('--kernel_sizes', type=int, nargs='+',
                        default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
    parser.add_argument('--lgag_ks', type=int,
                        default=3, help='Kernel size in LGAG')
    parser.add_argument('--activation_mscb', type=str,
                        default='relu6', help='activation used in MSCB: relu6 or relu')
    parser.add_argument('--no_dw_parallel', action='store_true', 
                        default=False, help='use this flag to disable depth-wise parallel convolutions')
    parser.add_argument('--concatenation', action='store_true', 
                        default=False, help='use this flag to concatenate feature maps in MSDC block')
    parser.add_argument('--no_pretrain', action='store_true', 
                        default=False, help='use this flag to turn off loading pretrained enocder weights')
    parser.add_argument('--pretrained_dir', type=str,
                        default='./pretrained_pth/pvt/', help='path to pretrained encoder dir')
    parser.add_argument('--supervision', type=str,
                    default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')    
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--test_batchsize', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--color_image', default=True)
    parser.add_argument('--augmentation', default=True)
    parser.add_argument('--train_path', type=str, default=f'./data/polyp/target/{dataset_name}/train/')
    parser.add_argument('--test_path', type=str, default=f'./data/polyp/target/{dataset_name}/')
    parser.add_argument('--train_save', type=str, default='') 
    parser.add_argument('--dataset_name', default=dataset_name)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--seed_base', type=int, default=2222)
    parser.add_argument('--wandb_project', default='EMCAD-polyp')
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    opt = parser.parse_args()
    opt.lr_schedule = 'CosineAnnealingLR'
    opt.lr_eta_min = 1e-6

    for split in ('train', 'val', 'test'):
        root = opt.train_path if split == 'train' else os.path.join(opt.test_path, split)
        for part in ('images', 'masks'):
            if not os.path.isdir(os.path.join(root, part)):
                parser.error(f'missing {split} {part}: {os.path.join(root, part)}')
    if not opt.no_pretrain and not os.path.isfile(os.path.join(opt.pretrained_dir, f'{opt.encoder}.pth')):
        parser.error(f'missing pretrained encoder: {os.path.join(opt.pretrained_dir, f"{opt.encoder}.pth")}')
    if opt.runs < 1:
        parser.error('--runs must be at least 1')
    if opt.dataset_name == 'Kvasir':
        for split in ('train', 'val', 'test'):
            list_path = Path('data/polyp/author_lists') / f'{split}_list_kvasir.txt'
            if not list_path.is_file():
                parser.error(f'missing author split list: {list_path}')
            expected = {Path(name).stem for name in list_path.read_text().splitlines() if name.strip()}
            image_path = Path(opt.train_path if split == 'train' else os.path.join(opt.test_path, split)) / 'images'
            actual = {path.stem for path in image_path.iterdir() if path.suffix.lower() in ('.jpg', '.png', '.jpeg')}
            if actual != expected:
                parser.error(f'{split} image membership differs from authors\' list')
        opt.author_split_sha256 = {split: file_sha256(Path('data/polyp/author_lists') / f'{split}_list_kvasir.txt')
                                   for split in ('train', 'val', 'test')}
    if not opt.no_pretrain:
        opt.encoder_weights_sha256 = file_sha256(os.path.join(opt.pretrained_dir, f'{opt.encoder}.pth'))
    if opt.wandb_mode != 'disabled':
        import wandb

    for run in range(1, opt.runs + 1):
        opt.seed = opt.seed_base + run - 1
        best = -1.0
        total_train_time = 0

        if opt.concatenation:
            aggregation = 'concat'
        else: 
            aggregation = 'add'
        
        if opt.no_dw_parallel:
            dw_mode = 'series'
        else: 
            dw_mode = 'parallel'

        timestamp = time.strftime('%H%M%S')
        run_id = (f"{opt.dataset_name}_{opt.encoder}_EMCAD_kernel_sizes_{opt.kernel_sizes}_dw_{dw_mode}_{aggregation}_lgag_ks_{opt.lgag_ks}_ef{opt.expansion_factor}_act_mscb_{opt.activation_mscb}_bs{opt.batchsize}_lr{opt.lr}_"
                      f"e{opt.epoch}_aug{opt.augmentation}_run{run}_seed{opt.seed}_t{timestamp}")
        run_id = run_id.replace('[', '').replace(']', '').replace(', ', '_')
        opt.train_save = f'./model_pth/{run_id}/'
        
        os.makedirs('logs', exist_ok=True)
        os.makedirs(opt.train_save, exist_ok=True)
        
        logging.basicConfig(filename=f'logs/train_log_{run_id}.log', level=logging.INFO, 
                            format='[%(asctime)s] %(message)s', force=True)

        if opt.wandb_mode != 'disabled':
            wandb.init(project=opt.wandb_project, name=run_id,
                       group='EMCAD-B0-Kvasir-paper', config=vars(opt), mode=opt.wandb_mode,
                       dir=os.path.abspath('.'),
                       tags=['EMCAD', opt.dataset_name, opt.encoder])
            wandb.define_metric('epoch')
            wandb.define_metric('train/*', step_metric='epoch')
            wandb.define_metric('val/*', step_metric='epoch')
            wandb.define_metric('test/*', step_metric='epoch')
            wandb.define_metric('best/*', step_metric='epoch')

        seed_everything(opt.seed)

        # Build model
        #model = EMCADNet(dw_parallel=dw_parallel, expansion_factor=expansion_factor, add=add, kernel_sizes=kernel_sizes, att_ks=att_ks, activation=activation, encoder=encoder, pretrain=pretrain, head=head, bbox=False, cds=False) # head='SAH'
        model = EMCADNet(num_classes=1, kernel_sizes=opt.kernel_sizes, expansion_factor=opt.expansion_factor, dw_parallel=not opt.no_dw_parallel, add=not opt.concatenation, lgag_ks=opt.lgag_ks, activation=opt.activation_mscb, encoder=opt.encoder, pretrain= not opt.no_pretrain, pretrained_dir=opt.pretrained_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)'''

        model.to(device)

        print(f"Encoder: {opt.encoder} | Decoder: EMCAD")
        param_count = sum(p.numel() for p in model.parameters())
        logging.info(f'Parameters: {param_count}')
        if opt.wandb_mode != 'disabled':
            wandb.run.summary['parameters'] = param_count
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=opt.lr_eta_min)

        train_loader = get_loader(
            image_root=f'{opt.train_path}/images/', gt_root=f'{opt.train_path}/masks/',
            batchsize=opt.batchsize, trainsize=opt.img_size, 
            shuffle=True, augmentation=opt.augmentation, split='train', color_image=opt.color_image,
            generator=torch.Generator().manual_seed(opt.seed)
        )

        for epoch in range(1, opt.epoch + 1):
            metrics = train(train_loader, model, optimizer, epoch, opt, run_id)
            if opt.wandb_mode != 'disabled':
                wandb.log({'epoch': epoch, **metrics})
            scheduler.step()
        best_path = os.path.join(opt.train_save, f"{run_id}-best.pth")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test_dice_at_best_val, test_iou_at_best_val, _ = test(model, opt.test_path, 'test', opt)
        logging.info(f'Final test at best validation checkpoint: Dice: {test_dice_at_best_val:.4f}, IoU: {test_iou_at_best_val:.4f}')
        if opt.wandb_mode != 'disabled':
            wandb.log({'epoch': opt.epoch, 'test/dice': test_dice_at_best_val, 'test/iou': test_iou_at_best_val})
        # FINAL SUMMARY
        
        summary = (f"\n{'='*40}\nFINAL RESULTS: {run_id}\n"
                   f"Best Val Dice: {best:.4f}\n"
                   f"Test Dice at Best Val: {test_dice_at_best_val:.4f}\n"
                   f"Test IoU at Best Val: {test_iou_at_best_val:.4f}\n"
                   f"Total Train Time: {total_train_time:.2f}s\n{'='*40}")
        print(summary)
        logging.info(summary)
        if opt.wandb_mode != 'disabled':
            wandb.run.summary['best_val_dice'] = best
            wandb.run.summary['test_dice_at_best_val'] = test_dice_at_best_val
            wandb.run.summary['test_iou_at_best_val'] = test_iou_at_best_val
            wandb.run.summary['total_train_seconds'] = total_train_time
            wandb.finish()
