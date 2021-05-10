# from models.pvt import pvt_tiny, pvt_small, pvt2048_small
# from models.hmr import HMR
# import torch
#
# # model = pvt_small()
# # model = HMR()
# model = pvt2048_small()
# img = torch.zeros([1, 3, 224, 224])
# out = model(img)
# t = 0


import torch
from utils.pose_utils import reconstruction_error
a = torch.randn([2, 14, 3]).numpy()
b = torch.randn([2, 14, 3]).numpy()
e = reconstruction_error(a, b, None, ['tran', 'rot'])
print(e)