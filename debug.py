from models.pvt import pvt_tiny, pvt_small, pvt2048_small
from models.hmr import HMR
from models.pvt_impr1 import pvt_small_impr1_peg
from models.my_pvt9 import mypvt9_small
import torch

# model = pvt_small()
# model = HMR()
# model = pvt2048_small()
# model = pvt_small_impr1_peg()
model = mypvt9_small()
img = torch.zeros([1, 3, 224, 224])
out = model(img)
t = 0


import torch
from utils.pose_utils import reconstruction_error
# a = torch.randn([2, 14, 3]).numpy()
# b = torch.randn([2, 14, 3]).numpy()
# e = reconstruction_error(a, b, None, ['tran', 'rot'])
# print(e)


import torch
B, N, C = 2, 3, 4
x = torch.rand(B, N, C, requires_grad=True)
pose_layer = torch.nn.Conv1d(N*C, N*9, kernel_size=1, groups=N)
rotmat = x
rotmat = rotmat.permute(0, 2, 1).reshape(B, -1, 1)  # (B, C*N, 1)
rotmat = pose_layer(rotmat)  # (B, 9*N, 1)
rotmat = rotmat.reshape(B, 9, N).reshape(B, 3, 3, N)
rotmat = rotmat.permute(0, 3, 1, 2)  # (B, N, 3, 3)

t = rotmat[0, 0].sum()
t.backward()
# rotmat = rotmat.reshape(-1, 3, 3).contiguous()


import torch
B, N, C = 2, 3, 4
x = torch.rand(B, N, C, requires_grad=True)
w = torch.rand(N, C, 9)