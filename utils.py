from torchvision.utils import save_image
import torch.nn as nn
import numpy as np
import config
import random
import torch
import math
import os

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_some_examples(gen, val_loader, epoch, folder):
    low_image_rgb, normal_image_rgb, _ = next(iter(val_loader))
    low_image_rgb, normal_image_rgb = low_image_rgb.to(config.DEVICE), normal_image_rgb.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        _, _, _, _, normal_pred = gen(low_image_rgb)
        save_image(normal_pred, folder + f"/y_recon_{epoch}.png")
        save_image(low_image_rgb, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(normal_image_rgb, folder + f"/label_{epoch}.png")
    gen.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def getFinalImage(input_image, atmosphere_light, refined_tranmission_map):
    """
    input_image: (1x3x256x256) = NCHW
    atmosphere_light: (1x3) = NC
    refined_tranmission_map: (1x1x256x256) = NCHW
    """
    refined_tranmission_map_broadcasted = torch.broadcast_to(refined_tranmission_map, (refined_tranmission_map.shape[0], 3, refined_tranmission_map.shape[2], refined_tranmission_map.shape[3]))
    refined_tranmission_map_broadcasted = refined_tranmission_map_broadcasted.permute(0, 2, 3, 1)
    input_image = input_image.permute(0, 2, 3, 1)
    refined_image = torch.empty(size=input_image.shape, dtype=input_image.dtype, device=input_image.device)
    for batch in range(input_image.shape[0]):
        # breakpoint()
        refined_image[batch, :, :, :] = (input_image[batch] - (1.0 - refined_tranmission_map_broadcasted[batch]) * atmosphere_light[batch]) / (torch.where(refined_tranmission_map_broadcasted[batch] < config.TMIN, config.TMIN, refined_tranmission_map_broadcasted[batch]))
    return ((refined_image - refined_image.min())/(refined_image.max() - refined_image.min())).permute(0, 3, 1, 2)

def getTransmissionMap(input_image, atmosphere_light, dark_channel_prior, bright_channel_prior, initial_transmission_map):
    """
    input_image: (1x3x256x256) = NCHW
    atmosphere_light: (1x3) = NC
    dark_channel_prior: (1x1x256x256) = NCHW
    bright_channel_prior: (1x1x256x256) = NCHW
    initial_transmission_map: (1x1x256x256) = NCHW
    """
    img = (1 - input_image) / (1 - atmosphere_light[:, :, None, None] + 1e-6)
    dark_channel_transmissionmap, _ = getIlluminationChannel(img)
    dark_channel_transmissionmap = dark_channel_transmissionmap
    dark_channel_transmissionmap = 1.0 - config.OMEGA * dark_channel_transmissionmap
    corrected_transmission_map = initial_transmission_map
    difference_channel_prior = bright_channel_prior - dark_channel_prior
    indices = difference_channel_prior < config.ALPHA
    corrected_transmission_map[indices] = dark_channel_transmissionmap[indices] * initial_transmission_map[indices]
    return torch.abs(corrected_transmission_map)

def getInitialTransmissionMap(atmosphere_light, bright_channel_prior):
    """
    atmosphere_light: (1x3) = NC
    bright_channel_prior: (1x1x256x256) = NCHW
    initial_transmission_map: (1x1x256x256) = NCHW
    """
    initial_transmission_map = (bright_channel_prior - torch.max(atmosphere_light)) / (1.0 - torch.max(atmosphere_light))
    return (initial_transmission_map - torch.min(initial_transmission_map))/(torch.max(initial_transmission_map) - torch.min(initial_transmission_map))

def getGlobalAtmosphereLight(input_image, bright_channel_prior, probability=0.1):
    """
    input_image: (1x3x256x256) = NCHW
    bright_channel_prior: (1x1x256x256) = NCHW
    atmosphere_light: (1x3) = NC
    """
    flattened_image = input_image.view(input_image.shape[0], input_image.shape[1], input_image.shape[2] * input_image.shape[3])
    flattened_bright_channel_prior = bright_channel_prior.view(bright_channel_prior.shape[0], bright_channel_prior.shape[2]*bright_channel_prior.shape[3])
    index = torch.argsort(flattened_bright_channel_prior, dim=-1, descending=True)[:, :int(input_image.shape[2] * input_image.shape[3] * probability)]
    atmosphere_light = torch.zeros((input_image.shape[0], 3), device="cuda")
    for i in range(input_image.shape[0]):
        atmosphere_light[i] = flattened_image[i, :, index].mean(axis=(1, 2))
    return atmosphere_light

def getIlluminationChannel(input_image):
    """
    input_image: (1x3x256x256) = NCHW
    dark_channel_prior: (1x1x256x256) = NCHW
    bright_channel_prior: (1x1x256x256) = NCHW
    """
    maxpool = nn.MaxPool3d((3, config.CHANNEL_PRIOR_KERNEL, config.CHANNEL_PRIOR_KERNEL), stride=(1, 1, 1), padding=(0, config.CHANNEL_PRIOR_KERNEL // 2, config.CHANNEL_PRIOR_KERNEL // 2))
    bright_channel_prior = maxpool(input_image)
    dark_channel_prior = maxpool(0.0 - input_image)
    
    return -dark_channel_prior, bright_channel_prior

def getRefinedImage(low_image_rgb, normal_image_rgb):
    low_image_rgb = low_image_rgb.to("cuda")
    normal_image_rgb = normal_image_rgb.to("cuda")
    input_image = (low_image_rgb + normal_image_rgb) / 2.0
    dark_channel_prior, bright_channel_prior = getIlluminationChannel(input_image)
    atmosphere_light = getGlobalAtmosphereLight(input_image, bright_channel_prior)
    initial_transmission_map = getInitialTransmissionMap(atmosphere_light, bright_channel_prior)
    transmission_map = getTransmissionMap(input_image, atmosphere_light, dark_channel_prior, bright_channel_prior, initial_transmission_map)
    refined_image = getFinalImage(low_image_rgb, atmosphere_light, transmission_map)
    return transmission_map, atmosphere_light, refined_image

# psnr
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
