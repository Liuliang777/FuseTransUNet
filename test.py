import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from evalution_segmentaion import eval_semantic_segmentation
from dataset import CamvidDataset
import cfg
from unet.unet_model import UNet

##########用GPU\CPU
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
# device = t.device('cpu')

#########初始化batch、miou
BATCH_SIZE = 1
miou_list = [0]

###########测试集数据集
Cam_test = CamvidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)
test_data = DataLoader(Cam_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

##########网络实例初始化
net = UNet(n_channels=3, n_classes=2)
net.eval()
net.to(device)

#########加载预训练模型
net.load_state_dict(t.load('./logs/49.pth'))


###############定义loss\miou\acc\mpa\error
train_acc = 0
train_miou = 0
train_class_acc = 0
train_mpa = 0
error = 0

######################开始抽取batch个img、label测试
for i, sample in enumerate(test_data):

	#######获取datalabel并转为tensor形式
	data = Variable(sample['img']).to(device)
	label = Variable(sample['label']).to(device)

	###########网络预测
	out = net(data)
	out = F.log_softmax(out, dim=1)
	###########

	pre_label = out.max(dim=1)[1].data.cpu().numpy()
	pre_label = [i for i in pre_label]

	true_label = label.data.cpu().numpy()
	true_label = [i for i in true_label]



###############以下是计算metrix\acc\miou\mpa\并在测试完成后展示结果（到最后）
	eval_metrix = eval_semantic_segmentation(pre_label, true_label)
	train_acc = eval_metrix['mean_class_accuracy'] + train_acc
	train_miou = eval_metrix['miou'] + train_miou
	train_mpa = eval_metrix['pixel_accuracy'] + train_mpa
	if len(eval_metrix['class_accuracy']) < 2:
		eval_metrix['class_accuracy'] = 0
		train_class_acc = train_class_acc + eval_metrix['class_accuracy']
		error += 1
	else:
		train_class_acc = train_class_acc + eval_metrix['class_accuracy']

	print(eval_metrix['class_accuracy'], '================', i)


epoch_str = ('test_acc :{:.5f} ,test_miou:{:.5f}, test_mpa:{:.5f}, test_class_acc :{:}'.format(train_acc /(len(test_data)-error),
															train_miou/(len(test_data)-error), train_mpa/(len(test_data)-error),
															train_class_acc/(len(test_data)-error)))

# if train_miou/(len(test_data)-error) > max(miou_list):
miou_list.append(train_miou/(len(test_data)-error))
print(epoch_str+'==========last')