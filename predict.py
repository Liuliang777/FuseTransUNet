from typing import List
import pandas as pd
import numpy as np
import torch
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from dataset1 import CamvidDataset
import cfg
from unet.unet_model import UNet
from evalution_segmentaion import eval_semantic_segmentation
from time import time
from cv2 import getTickCount,getTickFrequency
from utils.transunet import TransUNet
from torchcam.methods import GradCAM, GradCAMpp, ScoreCAM
from torchcam.utils import overlay_mask
start_time = time()
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
# device = t.device('cpu')


##############读取测试集并放入dataloader

Cam_test = CamvidDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Cam_test, batch_size=1, shuffle=True, num_workers=0)

from unet.Base_Unet_Vgg_Official import Base_Unet_Vgg_Official
##########网络实例初始化
net = TransUNet(img_dim=128,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=2)
# from unet.u import Unet
# net = Unet(3,2)
net.load_state_dict(t.load("./logs/t/49.pth",map_location='cpu'), strict=False)
net.cuda()
net.eval()

################读取label的颜色、名字
pd_label_color = pd.read_csv('./CamVid/class_dict.csv', sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
#########将上面读取的放入列表里
for i in range(num_class):
	tmp = pd_label_color.iloc[i]
	color = [tmp['r'], tmp['g'], tmp['b']]
	colormap.append(color)

cm = np.array(colormap).astype('uint8')

dir = "./result_pics/"

# miou_acc = 0
# pixel_acc = 0
# for i, sample in enumerate(test_data):
# 	valImg = sample['img'].to(device)
# 	valLabel = sample['label'].long().to(device)
# 	name = sample['name']
# 	# name=list(name)
# 	# print(name)
#
#
# 	out = net(valImg)
# 	out = F.log_softmax(out, dim=1)
# 	pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
# 	pre = cm[pre_label]
# 	pre1 = Image.fromarray(pre)
# 	pre1.save(dir + str(name)+'.png')
# 	print('Done')
#
# 	pre_label = out.max(dim=1)[1].data.cpu().numpy()
# 	pre_label = [i for i in pre_label]
# 	true_label = valLabel.data.cpu().numpy()
# 	true_label = [i for i in true_label]
#
# 	pre_metrics = eval_semantic_segmentation(pre_label, true_label)
# 	pixel_acc = pre_metrics['pixel_accuracy'] + pixel_acc
# 	miou_acc = pre_metrics['miou'] + miou_acc
#
# exec_time = (getTickCount() - exec_time) / getTickFrequency()
# print('FPS:{}'.format(exec_time))
# miou_acc = ('%f' % (miou_acc / len(test_data)))
# pixel_acc = ('%f' % (pixel_acc / len(test_data)))
# print('miou:{}\npixel_acc:{}\nmean_pixel_acc:{}'.format(miou_acc, pixel_acc))
miou_acc = 0
pixel_acc = 0
cam_extractor = ScoreCAM(net, target_layer="decoder")  # 根据torchcam库修改即可
index = 0
###########开始做predict
for i, sample in enumerate(test_data):

	import cv2
	#加载img、label、name
	valImg = sample['img'].to(device)
	valLabel = sample['label'].long().to(device)
	name = sample['name']
	# name=list(name)
	# print(name)


#######开始predict
	out = net(valImg)
	out = F.log_softmax(out, dim=1)



	###############预测得到label、img并保存
	pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
	pre = cm[pre_label]
	pre1 = Image.fromarray(pre)
	pre1.save(dir + str(name) + '.png')
	print('Done')


################评估预测label和真实label的差距、做分割
	pre_label = out.max(dim=1)[1].data.cpu().numpy()
	pre_label = [i for i in pre_label]
	true_label = valLabel.data.cpu().numpy()
	true_label = [i for i in true_label]

	pre_metrics = eval_semantic_segmentation(pre_label, true_label)

	# pixel_acc = pre_metrics['pixel_accuracy'] + pixel_acc
	# miou_acc = pre_metrics['iou'] + miou_acc
#
# end_time = time()
# print('平均推理时间为:{}'.format((end_time - start_time) / len(test_data)))
# miou_acc = ('%f' % (miou_acc / len(test_data)))
# pixel_acc = ('%f' % (pixel_acc / len(test_data)))
# print('pixel_acc:{}\nmiou_acc:{}'.format( pixel_acc, miou_acc))