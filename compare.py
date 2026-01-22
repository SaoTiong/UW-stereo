import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image


DEVICE = 'cuda'
IMAGE_PATH_L = "你的测试图片_左.jpg"  # 替换这里
IMAGE_PATH_R = "你的测试图片_右.jpg"  # 替换这里

# 2. 定义加载模型的函数
def load_model(checkpoint_path):
    args = argparse.Namespace(
        hidden_dims=[128]*3, context_norm='instance', corr_implementation='alt', 
        shared_backbone=False, corr_levels=4, corr_radius=4, n_downsample=2,
        n_gru_layers=3, slow_fast_gru=False, mixed_precision=True
    )
    model = RAFTStereo(args).to(DEVICE)
    model = torch.nn.DataParallel(model)
    
    # 加载权重 (处理 DataParallel 的 'module.' 前缀)
    ckpt = torch.load(checkpoint_path)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    return model

# 3. 准备图像
imgL = np.array(Image.open(IMAGE_PATH_L)).astype(np.uint8)
imgR = np.array(Image.open(IMAGE_PATH_R)).astype(np.uint8)
imgL = torch.from_numpy(imgL).permute(2, 0, 1).float()[None].to(DEVICE)
imgR = torch.from_numpy(imgR).permute(2, 0, 1).float()[None].to(DEVICE)

padder = InputPadder(imgL.shape, divis_by=32)
imgL_pad, imgR_pad = padder.pad(imgL, imgR)

# 4. 加载两个模型
print("Loading Pretrained Model (RVC)...")
model_old = load_model("models/iraftstereo_rvc.pth") # 指向你的原始模型

print("Loading Finetuned Model (Yours)...")
model_new = load_model("checkpoints/50000_uwraftstereo.pth") # 指向你的新模型

# 5. 推理
print("Running Inference...")
with torch.no_grad():
    _, flow_old = model_old(imgL_pad, imgR_pad, iters=32, test_mode=True)
    _, flow_new = model_new(imgL_pad, imgR_pad, iters=32, test_mode=True)

# 去除 Padding
disp_old = padder.unpad(flow_old)[0,0].cpu().numpy() * -1.0 # 取负值转视差
disp_new = padder.unpad(flow_new)[0,0].cpu().numpy() * -1.0

# 6. 计算差异 (Difference Map)
diff = np.abs(disp_old - disp_new)

# 7. 可视化
plt.figure(figsize=(20, 10))

# 原图
plt.subplot(2, 2, 1)
plt.title("Left Image (Underwater)")
plt.imshow(np.array(Image.open(IMAGE_PATH_L)))
plt.axis('off')

# 差异热力图
plt.subplot(2, 2, 2)
plt.title("Difference Heatmap (Where did it change?)")
# vmin=0, vmax=5 表示只关注 0-5 像素的变化，高亮差异
plt.imshow(diff, cmap='jet', vmin=0, vmax=5) 
plt.colorbar(label='Pixel Difference')
plt.axis('off')


plt.subplot(2, 2, 3)
plt.title("Pre-trained RVC Model")
plt.imshow(disp_old, cmap='magma')
plt.axis('off')


plt.subplot(2, 2, 4)
plt.title("Finetuned UW Model (50k)")
plt.imshow(disp_new, cmap='magma')
plt.axis('off')

plt.tight_layout()
plt.savefig("comparison_result.png")
print("Saved comparison to comparison_result.png")
plt.show()