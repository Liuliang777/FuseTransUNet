import torch as t
import torch.nn as nn
import torch.nn.functional as F
import time
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import CamvidDataset
from evalution_segmentaion import eval_semantic_segmentation

from unet.aunet import AttU_Net
from utils.transunet3 import TransUNet
from unet.u import Unet
import cfg

#######设置GPU#########
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

##########如果没有GPU则用CPU#########
# device = t.device('cpu')


#########设置训练集、验证集及各自标签的根路径，裁剪size##############
Cam_train = CamvidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Cam_val = CamvidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)
########用Dataloader加载数据集######################
train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
val_data = DataLoader(Cam_val, batch_size=1, shuffle=True, num_workers=0)

#########Unet网络实例初始化定义损失函数、优化器#######################

fcn = TransUNet(img_dim=128,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=2)
fcn = Unet(3,2)
fcn = fcn.to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)

start_time = time.time()
##############定义训练函数##################################
def train(model):
    #######开始训练##########
    best = [0]
    net = model.train()
    # 训练轮次，每50轮，改变学习率（规则：*0.5）#################
    for epoch in range(cfg.EPOCH_NUMBER):
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        #######初始化loss、acc、miou#################
        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0


        # 训练批次 #################从traindata抽取batch个训练集
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample['img'].to(device))   # [4, 3, 352, 480]
            img_label = Variable(sample['label'].to(device))    # [4, 352, 480]
            # 训练
            ######前向传播流程
            out = net(img_data)     # [4, 12, 352, 480]
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 模型评估：metrix\acc\miou等指标
            pre_label = out.max(dim=1)[1].data.cpu().numpy()    # (4, 352, 480)
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()   # (4, 352, 480)
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            # train_class_acc += eval_metrix['class_accuracy']

            
###############在网络训练时打印batch、loss
            print('|batch[{}/{}]|batch_loss {: .8f}|'.format(i + 1, len(train_data), loss.item()))
            file_handle2=open('train_loss.csv',mode='a+')
            loss = ('%f' % (loss))
  
            file_handle2.write(loss +'\n')
            file_handle2.close()

###############计算一个epoch的平均acc、miou
        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            # train_class_acc / len(train_data),
        )

#############保存每个epoch最好的模型、路径。
        #
        #if max(best) <= train_miou / len(train_data):
            #best.append(train_miou / len(train_data))
        print(metric_description)
        t.save(net.state_dict(), './logs/{}.pth'.format(epoch))


        ##########评估模型 （验证集）以下均与上面训练过程相同。
        net = model.eval()
        eval_loss = 0
        eval_acc = 0
        eval_miou = 0
        eval_class_acc = 0
        eval_pixel_acc = 0

        prec_time = datetime.now()
        for j, sample in enumerate(val_data):
            valImg = Variable(sample['img'].to(device))
            valLabel = Variable(sample['label'].long().to(device))

            out = net(valImg)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, valLabel)
            eval_loss = loss.item() + eval_loss
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = valLabel.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrics = eval_semantic_segmentation(pre_label, true_label)
            eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
            eval_miou = eval_metrics['miou'] + eval_miou
            eval_pixel_acc = eval_metrics['pixel_accuracy'] + eval_pixel_acc
        # eval_class_acc = eval_metrix['class_accuracy'] + eval_class_acc

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prec_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

        val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:} \n|Valid Pixel Acc|:{:.5f}'.format(
            eval_loss / len(train_data),
            eval_acc / len(val_data),
            eval_miou / len(val_data),
            eval_class_acc / len(val_data),
            eval_pixel_acc / len(val_data)))

        print(val_str)
        print(time_str)
        file_handle2=open('miou.csv',mode='a+')
        miou = ('%f' % (train_miou / len(train_data)))
        
        evalmiou = ('%f' % (eval_miou / len(val_data)))



  ##############将训练日志写入csv文件
        file_handle2.write(str(epoch + 1)+','+miou+','+evalmiou +'\n')
        file_handle2.close()

        file_handle3=open('pixel_error.csv',mode='a+')
        pixel_error = ('%f' % (1 - (eval_pixel_acc / len(val_data))))
        file_handle3.write(pixel_error + '\n')
        file_handle3.close()
#
###########模型评估函数
def evaluate(model):

    ############模型初始化、定义loss\miou\acc\等指标
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_class_acc = 0


################从valdata里加载一个batch的数据
    prec_time = datetime.now()
    for j, sample in enumerate(val_data):

        ###加载img和label并转换为tensor
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))



########网络验证的流程。forward过程。
        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

##########得到true label
        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]
###########模型评估指标
        eval_metrics = eval_semantic_segmentation(pre_label, true_label)
        eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
        eval_miou = eval_metrics['miou'] + eval_miou
    # eval_class_acc = eval_metrix['class_accuracy'] + eval_class_acc


##########以下均为验证过程中打印、写入评估日志。
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)

    val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:}'.format(
        eval_loss / len(train_data),
        eval_acc / len(val_data),
        eval_miou / len(val_data),
        eval_class_acc / len(val_data)))
    print(val_str)
    print(time_str)


if __name__ == "__main__":
    train(fcn)
    evaluate(fcn)
    end_time = time.time()
    print("训练时间为：{}".format(end_time - start_time))

