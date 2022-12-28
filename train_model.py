from utils import (save_checkpoint, load_checkpoint, calculate_psnr, 
                getRefinedImage, getFinalImage, seed_everything, 
                gradient_penalty)
                
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models import LIENet, Discriminator
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

# gen, critic, train_loader, opt_gen, opt_critic, loss_fn, train_writer
def train_fn(gen, critic, train_loader, optimizer_gen, optimizer_critic, loss_fn, writer):
    loop = tqdm(train_loader)
    curr_psnr = 0
    global train_step

    gen.train()
    critic.train()

    for batch_idx, (low_image_rgb, normal_image_rgb, normal_image_gray) in enumerate(loop):
        low_image_rgb = low_image_rgb.to(device=config.DEVICE)
        normal_image_rgb = normal_image_rgb.to(device=config.DEVICE)
        normal_image_gray = normal_image_gray.to(device=config.DEVICE)

        # get ground truth of trasmission map and atmospheric light
        orig_transmission_map, orig_atmosphere_light, refined_image = getRefinedImage(low_image_rgb, normal_image_rgb)

        # for _ in range(config.CRITIC_ITERATIONS):
        pred_gray_scale, pred_transmission_map, pred_atmosphere_light, pred_refined_map, pred_normal_image_rgb = gen(low_image_rgb)
            # fake = torch.cat([low_image_rgb, refined_image, pred_normal_image_rgb], axis=1)
            # real = torch.cat([low_image_rgb, pred_refined_map, normal_image_rgb], axis=1)

            # critic_real = critic(real).reshape(-1)
            # critic_fake = critic(fake).reshape(-1)

            # gp = gradient_penalty(critic, real, fake, device=config.DEVICE)
            # loss_critic = (
            #     -(torch.mean(critic_real) - torch.mean(critic_fake)) + config.LAMBDA_GP * gp
            # )
            # critic.zero_grad()
            # loss_critic.backward(retain_graph=True)
            # optimizer_critic.step()

        # gen_fake = critic(fake)
        # loss_critic = -torch.mean(gen_fake)
        total_loss, tm_l1, atmos_l1, out_l1 = loss_fn(normal_image_rgb, normal_image_gray, orig_transmission_map, orig_atmosphere_light, pred_normal_image_rgb, pred_gray_scale, pred_transmission_map, pred_atmosphere_light)
        # total_loss += loss_critic
    
        # backward
        gen.zero_grad()
        total_loss.backward()
        optimizer_gen.step()

        curr_psnr += calculate_psnr(normal_image_rgb.detach().cpu().numpy() * 255.0, pred_normal_image_rgb.detach().cpu().numpy() * 255.0)
        avg_psnr = curr_psnr / (batch_idx + 1)
        
        # update tqdm loop
        loop.set_postfix(
            # coarse_loss=coarse_loss.item(),
            total_loss=total_loss.item(),
            tm_l1=tm_l1.item(),
            atmos_l1=atmos_l1.item(),
            # tm_dice=tm_dice.item(),
            out_l1=out_l1.item(),
            # out_ssim=out_ssim.item(),
            # out_color=out_color.item(),
            # out_tv=out_tv.item(),
            psnr=avg_psnr
        )

        # tensorboard logs
        writer.add_scalar("Training loss", total_loss, global_step=train_step)
        writer.add_scalar("Train Transmission Map Loss", tm_l1, global_step=train_step)
        # writer.add_scalar("Train Atmospheric Light Loss", atmos_l1, global_step=train_step)
        writer.add_scalar("Train Recon Loss", out_l1, global_step=train_step)
        # writer.add_scalar("Train Color Loss", out_color, global_step=train_step)
        writer.add_scalar("Train PSNR", avg_psnr, global_step=train_step)
        train_step+=1
    return avg_psnr

def val_fn(gen, test_loader, writer):

    gen.eval()
    loop = tqdm(test_loader)
    curr_psnr = 0
    ref_psnr = 0
    global val_step

    for batch_idx, (low_image_rgb, normal_image_rgb, normal_image_gray) in enumerate(loop):
        low_image_rgb = low_image_rgb.to(device=config.DEVICE)
        normal_image_rgb = normal_image_rgb.to(device=config.DEVICE)
        normal_image_gray = normal_image_gray.to(device=config.DEVICE)
        
        # get ground truth of trasmission map and atmospheric light
        orig_transmission_map, _, orig_refined_map = getRefinedImage(low_image_rgb, normal_image_rgb)

        with torch.no_grad():
            # forward
            pred_gray_scale, pred_transmission_map, pred_atmosphere_light, pred_refined_map, pred_normal_image_rgb = gen(low_image_rgb)
            # input_image = (low_image_rgb + normal_image_rgb) / 2.0
            # pred_refined_map = getFinalImage(input_image, pred_atmosphere_light, pred_transmission_map)
        
        # originals
        low_img_grid = torchvision.utils.make_grid(low_image_rgb)
        orig_normal_img_grid = torchvision.utils.make_grid(normal_image_rgb)
        orig_transmission_map_grid = torchvision.utils.make_grid(orig_transmission_map)
        orig_refined_map_grid = torchvision.utils.make_grid(orig_refined_map)
        orig_gray_scale_grid = torchvision.utils.make_grid(normal_image_gray)

        writer.add_image("low light images", low_img_grid, global_step=val_step)
        writer.add_image("original normal light images", orig_normal_img_grid, global_step=val_step)
        writer.add_image("original transmission map images", orig_transmission_map_grid, global_step=val_step)
        writer.add_image("original refined map images", orig_refined_map_grid, global_step=val_step)
        writer.add_image("original gray_scale images", orig_gray_scale_grid, global_step=val_step)

        # preds
        pred_normal_img_grid = torchvision.utils.make_grid(pred_normal_image_rgb)
        pred_transmission_map_grid = torchvision.utils.make_grid(pred_transmission_map)
        pred_refined_map_grid = torchvision.utils.make_grid(pred_refined_map)  
        pred_gray_scale_grid = torchvision.utils.make_grid(pred_gray_scale)

        writer.add_image("predicted normal light images", pred_normal_img_grid, global_step=val_step)
        writer.add_image("predicted transmission map images", pred_transmission_map_grid, global_step=val_step)
        writer.add_image("predicted refined map images", pred_refined_map_grid, global_step=val_step)
        writer.add_image("predicted gray_scale images", pred_gray_scale_grid, global_step=val_step)
        
        curr_psnr += calculate_psnr(normal_image_rgb.detach().cpu().numpy() * 255.0, pred_normal_image_rgb.detach().cpu().numpy() * 255.0)
        avg_psnr = curr_psnr / (batch_idx + 1)

        # ref_psnr += calculate_psnr(normal_image_rgb.detach().cpu().numpy() * 255.0, pred_refined_map.detach().cpu().numpy() * 255.0)
        # avg_psnr = curr_psnr / (batch_idx + 1)
        # avg_reg_psnr = ref_psnr / (batch_idx + 1)
        
        # update tqdm loop
        loop.set_postfix(
            psnr=avg_psnr,
            # ref_psnr=avg_reg_psnr
        )

        # tensorboard logs
        writer.add_scalar("valid PSNR", avg_psnr, global_step=val_step)
        val_step+=1
    return avg_psnr

def main():
    seed_everything(42)

    gen = LIENet().to(config.DEVICE)
    critic = Discriminator(9, 16).to(config.DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE)
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE)
    loss_fn = Loss()
    
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
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
    # iters = len(train_loader)
    
    # Visualize model in TensorBoard
    images, _, _ = next(iter(train_loader))
    train_writer.add_graph(gen, images.to(config.DEVICE))
    train_writer.close()
    
    best_psnr = 0.0
    best_epoch = 0
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_DIR, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_DIR, critic, opt_critic, config.LEARNING_RATE,
        )
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch : {epoch}/{config.NUM_EPOCHS}")
        train_psnr = train_fn(
            gen, critic, train_loader, opt_gen, opt_critic, loss_fn, train_writer
        )
        test_psnr = val_fn(
            gen, val_loader, val_writer
        )
        print('Training PSNR: ', train_psnr)

        if test_psnr > best_psnr:
            best_psnr = test_psnr
            best_epoch = epoch
            print(f"Saving checkpoint with best psnr of {test_psnr} at epoch {epoch}")
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN_DIR)
            save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC_DIR)
        else:
            print(f"Saved best checkpoint with best psnr of {best_psnr} at epoch {best_epoch}")
            
        # save_some_examples(gen, val_loader, epoch, folder=config.RESULTS_DIR)


if __name__ == "__main__":
    main()
