import numpy as np
import torch
from einops import rearrange
from PIL import Image
import torchvision
import torchvision.transforms as transforms

transform_PIL = transforms.Compose([
    transforms.ToPILImage(),
])

color_dict = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "blue": (5, 39, 175),
}

# 颜色列表
# 红，绿，黄，蓝
# 紫，青，
# 灰色，冷灰色，石板灰，暖灰色
# 香蕉色，镉黄，dougello，forum gold
# 金黄色，黄花色
color_list = [
    (255, 0, 0), (0, 255, 0), (255, 255, 0), (5, 39, 175),
    (255,0,255), (0,255,255), 
    (192,192,192), (128,138,135), (112,128,105), (128,128,105),
    (227,207,87), (255,153,18), (235,142,85), (255,227,132),
    (255,215,0), (218,165,105)
]
# https://blog.csdn.net/pinaby/article/details/2823366


# same function in torchvision.utils.save_image(normalize=True)
def image_normalize(tensor, value_range=None, scale_each=False):
    tensor = tensor.clone()
    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t, value_range)
    else:
        norm_range(tensor, value_range)
    
    return tensor

def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])
 
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255
 
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )
 
    return img_

# images, score_map: tensor
def build_score_image(images, score_map, low_color="blue", high_color="red", scaler=0.9):
    bs = images.size(0)

    low = Image.new('RGB', (images.size(-2), images.size(-1)), color_dict[low_color])
    high = Image.new('RGB', (images.size(-2), images.size(-1)), color_dict[high_color])

    for i in range(bs):
        image_i_pil = transform_PIL(image_normalize(images[i]))

        score_map_i_np = rearrange(score_map[i], "C H W -> H W C").cpu().detach().numpy()
        score_map_i_blend = Image.fromarray(
            np.uint8(high * score_map_i_np + low * (1 - score_map_i_np)))
        
        image_i_blend = Image.blend(image_i_pil, score_map_i_blend, scaler)

        if i == 0:
            blended_images = torchvision.transforms.functional.to_tensor(image_i_blend).unsqueeze(0)
        else:
            blended_images = torch.cat([
                blended_images, torchvision.transforms.functional.to_tensor(image_i_blend).unsqueeze(0)
            ], dim=0)
    return blended_images


if __name__ == "__main__":
    images = torch.zeros(1, 3, 256, 256)
    score_map = torch.tensor([[[
        [0.20, 0.07, 0.64, 0.09],
        [0.14, 0.12, 0.32, 0.02],
        [0.22, 0.97, 0.07, 0.07],
        [0.32, 0.37, 0.12, 0.53]
    ]]]).repeat_interleave(64, -1).repeat_interleave(64, -2)
    out = build_score_image(images, score_map, low_color="blue", high_color="red", scaler=0.9)
    torchvision.utils.save_image(out, "out.png")