from models.pvt import pvt_tiny
import torch

model = pvt_tiny()
img = torch.zeros([1, 3, 224, 224])
out = model(img)
t = 0
