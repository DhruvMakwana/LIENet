from utils import save_checkpoint, load_checkpoint, save_some_examples, calculate_psnr, getRefinedImage, seed_everything
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models import LIE_Prior
from dataset import LolDataset
import torch.optim as optim
from losses import Loss
from tqdm import tqdm
import torch.nn as nn
import torchvision
import config
import torch

train_step=0
val_step=0

def train_fn(model, train_loader, optimizer, loss_fn, writer):
    loop = tqdm(train_loader)
    curr_psnr = 0
    global train_step

    for batch_idx, (low_image_rgb, normal_image_rgb, normal_image_gray) in enumerate(loop):
        low_image_rgb = low_image_rgb.to(device=config.DEVICE)
        normal_image_rgb = normal_image_rgb.to(device=config.DEVICE)
        normal_image_gray = normal_image_gray.to(device=config.DEVICE)

        # get ground truth of trasmission map and atmospheric light
        orig_transmission_map, orig_atmosphere_light, refined_image = getRefinedImage(low_image_rgb, normal_image_rgb)

        # forward
        pred_gray, pred_transmission_map, pred_atmosphere_light, pred_normal_image_rgb = model(low_image_rgb)

        total_loss, gray_l1, gray_ssim, tm_l1, atmos_l1, out_l1 = loss_fn(normal_image_rgb, normal_image_gray, orig_transmission_map, orig_atmosphere_light, pred_normal_image_rgb, pred_gray, pred_transmission_map, pred_atmosphere_light)

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        curr_psnr += calculate_psnr(normal_image_rgb.detach().cpu().numpy() * 255.0, pred_normal_image_rgb.detach().cpu().numpy() * 255.0)
        avg_psnr = curr_psnr / (batch_idx + 1)
        
        # update tqdm loop
        loop.set_postfix(
            loss=total_loss.item(),
            gray_l1=gray_l1.item(),
            gray_ssim=gray_ssim.item(),
            tm_l1=tm_l1.item(),
            atmos_l1=atmos_l1.item(),
            out_l1=out_l1.item(),
            psnr=avg_psnr
        )

        # tensorboard logs
        writer.add_scalar("Training loss", total_loss, global_step=train_step)
        writer.add_scalar("Train Gray Image L1 loss", gray_l1, global_step=train_step)
        writer.add_scalar("Train Gray Image SSIM Loss", gray_ssim, global_step=train_step)
        writer.add_scalar("Train Transmission Map Loss", tm_l1, global_step=train_step)
        writer.add_scalar("Train Atmospheric Light Loss", atmos_l1, global_step=train_step)
        writer.add_scalar("Train Recon Loss", out_l1, global_step=train_step)
        writer.add_scalar("Train PSNR", avg_psnr, global_step=train_step)
        train_step+=1
    return avg_psnr

def val_fn(model, test_loader, writer):
    model.eval()
    loop = tqdm(test_loader)
    curr_psnr = 0
    global val_step

    for batch_idx, (low_image_rgb, normal_image_rgb, normal_image_gray) in enumerate(loop):
        low_image_rgb = low_image_rgb.to(device=config.DEVICE)
        normal_image_rgb = normal_image_rgb.to(device=config.DEVICE)
        normal_image_gray = normal_image_gray.to(device=config.DEVICE)

        # get ground truth of trasmission map and atmospheric light
        orig_transmission_map, _, _ = getRefinedImage(low_image_rgb, normal_image_rgb)

        with torch.no_grad():
            # forward
            pred_gray, pred_transmission_map, _, pred_normal_image_rgb = model(low_image_rgb)
        
        # originals
        low_img_grid = torchvision.utils.make_grid(low_image_rgb)
        orig_normal_img_grid = torchvision.utils.make_grid(normal_image_rgb)
        orig_gray_grid = torchvision.utils.make_grid(normal_image_gray)
        orig_transmission_map_grid = torchvision.utils.make_grid(orig_transmission_map)

        writer.add_image("low light images", low_img_grid, global_step=val_step)
        writer.add_image("original normal light images", orig_normal_img_grid, global_step=val_step)
        writer.add_image("original gray images", orig_gray_grid, global_step=val_step)
        writer.add_image("original transmission map images", orig_transmission_map_grid, global_step=val_step)

        # preds
        pred_normal_img_grid = torchvision.utils.make_grid(pred_normal_image_rgb)
        pred_gray_grid = torchvision.utils.make_grid(pred_gray)
        pred_transmission_map_grid = torchvision.utils.make_grid(pred_transmission_map)

        writer.add_image("predicted normal light images", pred_normal_img_grid, global_step=val_step)
        writer.add_image("predicted gray images", pred_gray_grid, global_step=val_step)
        writer.add_image("predicted transmission map images", pred_transmission_map_grid, global_step=val_step)
        
        curr_psnr += calculate_psnr(normal_image_rgb.detach().cpu().numpy() * 255.0, pred_normal_image_rgb.detach().cpu().numpy() * 255.0)
        avg_psnr = curr_psnr / (batch_idx + 1)
        
        # update tqdm loop
        loop.set_postfix(
            psnr=avg_psnr
        )

        # tensorboard logs
        writer.add_scalar("valid PSNR", avg_psnr, global_step=val_step)
        val_step+=1

    model.train()
    return avg_psnr

def main():
    seed_everything(42)
    model = LIE_Prior().to(config.DEVICE)
    loss_fn = Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    train_writer = SummaryWriter(config.TRAIN_LOGS_DIR)
    val_writer = SummaryWriter(config.VAL_LOGS_DIR)
    
    train_dataset = LolDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory = True,
    )
    
    val_dataset = LolDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Visualize model in TensorBoard
    images, _, _ = next(iter(train_loader))
    train_writer.add_graph(model, images.to(config.DEVICE))
    train_writer.close()
    
    best_psnr = 19.6
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_DIR, model, optimizer, config.LEARNING_RATE,
        )
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch : {epoch}/{config.NUM_EPOCHS}")
        train_psnr = train_fn(
            model, train_loader, optimizer, loss_fn, train_writer
        )
        test_psnr = val_fn(
            model, val_loader, val_writer
        )

        if test_psnr > best_psnr:
            best_psnr = test_psnr
            print(f"Saving checkpoint with best psnr of {test_psnr} at epoch {epoch}")
            save_checkpoint(model, optimizer, filename=config.CHECKPOINT_DIR)
        save_some_examples(model, val_loader, epoch, folder=config.RESULTS_DIR)


if __name__ == "__main__":
    main()