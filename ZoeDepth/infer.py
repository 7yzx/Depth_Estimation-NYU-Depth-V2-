import argparse
from pprint import pprint

import torch
load_path = '/home/whuai/Depth_Estimation-NYU-Depth-V2-/ZoeDepth/models/ZoeD_M12_N.pt'
# # Zoe_N
# model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# ZoeD_N
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)
##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# Local file
from PIL import Image
import matplotlib.pyplot as plt
image = Image.open("./input/1.jpg").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy


depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image
# import os
# plt.imshow(depth_numpy, cmap='gray')
# plt.axis('off')
# plt.savefig(os.path.join('./result1/' , f'vis_gt_{i:05}.jpg'),
#             bbox_inches='tight', pad_inches=0)
# plt.close()
# depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor


# Tensor
from zoedepth.utils.misc import pil_to_batched_tensor
X = pil_to_batched_tensor(image).to(DEVICE)
depth = zoe.infer(X)

# Save raw
# from zoedepth.utils.misc import save_raw_16bit
# fpath = "./result1/out.png"
# save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth)

# save colored output
fpath_colored = "./result/colored/output_colored.png"
Image.fromarray(colored).save(fpath_colored)