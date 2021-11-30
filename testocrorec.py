import sys
import re
import os.path
import matplotlib.pyplot as plt
import torch
import torch.jit
import webdataset as wds
import yaml
import PIL.Image
import numpy as np
from ocropus.ocrorec2 import ctc_decode

model = torch.jit.load("model.jit")
model.eval()
model.cuda()

assert os.path.exists(sys.argv[1]), sys.argv[1]
def goodtext(sample):
    return re.search(r"[a-zA-Z0-9_\-\.]{3,}", sample["txt"])
ds = wds.WebDataset(sys.argv[1]).decode("torchrgb").select(goodtext).rename(image="png;jpg;jpeg")
src = iter(ds)

plt.ion()
for sample in src:
    input = sample["image"]
    input = model.standardize(input)
    input = model.auto_resize(input)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(input.permute(1, 2, 0))
    plt.title(sample["txt"])
    plt.subplot(1, 2, 2)

    output = model.forward(input.unsqueeze(0).cuda()).softmax(1).cpu()
    decoded = torch.tensor(ctc_decode(output[0], sigma=1.0, threshold=0.9))
    print(sample.get("json"))
    plt.imshow(output[0].detach().numpy())
    plt.title(model.decode_str(decoded))
    plt.ginput(1)


# %%
