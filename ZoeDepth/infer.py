import argparse
from pprint import pprint

import torch
# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# Local file
from PIL import Image
image = Image.open("./input/1.jpg").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

# Tensor
from zoedepth.utils.misc import pil_to_batched_tensor
X = pil_to_batched_tensor(image).to(DEVICE)
depth = zoe.infer(X)

# Save raw
from zoedepth.utils.misc import save_raw_16bit
fpath = "./result/"
save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth)

# save colored output
fpath_colored = "./result/colored/output_colored.png"
Image.fromarray(colored).save(fpath_colored)