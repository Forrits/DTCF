import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vnet_sdf import VNet
from networks.ResNet34_sdf import Resnet34

from utils import ramps, losses
from dataloaders.paccer import Pancreas
# from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor, TwoStreamBatchSampler
from lib import transforms_for_rot, transforms_back_rot, transforms_for_noise, transforms_for_scale, \
    transforms_back_scale, postprocess_scale
import sys
from dataloaders.brats2019 import BraTS2019,RandomRotFlip,RandomCrop,ToTensor,TwoStreamBatchSampler
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='dataset', help='Name of Experiment')  # todo change dataset path
parser.add_argument('--exp', type=str, default="MCF_flod0", help='model_name')  # todo model name
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')

args = parser.parse_args()  # 解析

train_data_path = args.root_path
snapshot_path = "./2024.9.23/" + args.exp + "/dtcf"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
#patch_size = (112, 112, 80)
patch_size=(96,96,96)
T = 0.1
Good_student = 0  # 0: vnet 1:resnet


# 权值更新策略
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c - 1):
        temp_line = vec[:, i, :].unsqueeze(1)  # b 1 c
        star_index = i + 1
        rep_num = c - star_index
        repeat_line = temp_line.repeat(1, rep_num, 1)
        two_patch = vec[:, star_index:, :]
        temp_cat = torch.cat((repeat_line, two_patch), dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result, dim=1)
    return result


if __name__ == "__main__":

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(name='vnet'):  # 实例化模型
        # Network definition
        if name == 'vnet':
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        if name == 'resnet34':
            net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        return model


    model_vnet = create_model(name='vnet')
    model_resnet = create_model(name='resnet34')
#################################  LA数据级
    # db_train = LAHeart(base_dir=train_data_path,
    #                    split='train',
    #                    train_flod="train4.list",  # todo change training flod
    #                    common_transform=transforms.Compose([  # comon_transform公共变换
    #                        RandomCrop(patch_size),
    #                    ]),
    #                    sp_transform=transforms.Compose([
    #                        ToTensor(),
    #                    ]))
    # # 有标签的索引是【0，15】
    # labeled_idxs = list(range(16))  # todo set labeled num
    # # 无标签的索引是[16,79]
    # unlabeled_idxs = list(range(16, 80))  # todo set labeled num all_sample_num
    # labeled_idxs = list(range(12))           # todo set labeled num
    # #无标签的索引是[16,79]
    # unlabeled_idxs = list(range(12, 61))   

    
    
    
####################### bra数据集
    db_train = BraTS2019(base_dir='dataset/data',
                   split='train',
                   num=None,
                   transform=transforms.Compose([
                       RandomRotFlip(),
                       RandomCrop(patch_size),
                       ToTensor(),
                   ]))
    labeled_idxs,unlabeled_idxs = list(range(50)) , list(range(50, 250)) 
  
#####################这定义采样策略
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    vnet_optimizer = optim.SGD(model_vnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    resnet_optimizer = optim.SGD(model_resnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    star_epoch = -1
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    model_vnet.train()
    model_resnet.train()

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            print('epoch:{},i_batch:{}'.format(epoch_num, i_batch))
            volume_batch1, volume_label1 = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']
            v_input, v_label = volume_batch1.cuda(), volume_label1.cuda()
            r_input, r_label = volume_batch2.cuda(), volume_label2.cuda()
            # 无标记数据
            v_unlabeled_batch = v_input[labeled_bs:]
            r_unlabeled_batch = r_input[labeled_bs:]
            v_unlabeled_batch, r_unlabeled_batch = v_unlabeled_batch.cuda(), r_unlabeled_batch.cuda()
            inputs_v_noise = transforms_for_noise(v_unlabeled_batch)
            inputs_r_noise = transforms_for_noise(r_unlabeled_batch)
            inputs_v_noise, v_rot_mask, v_flip_mask = transforms_for_rot(inputs_v_noise)
            inputs_r_noise, r_rot_mask, r_flip_mask = transforms_for_rot(inputs_r_noise)
            v_outdis,v_outputs = model_vnet(v_input)
            r_outdis,r_outputs = model_resnet(r_input)
            ## calculate the supervised loss
            v_loss_seg = F.cross_entropy(v_outputs[:labeled_bs], v_label[:labeled_bs])
            v_outputs_soft = F.softmax(v_outputs, dim=1)
            v_loss_seg_dice = losses.dice_loss(v_outputs_soft[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] == 1)
            r_loss_seg = F.cross_entropy(r_outputs[:labeled_bs], r_label[:labeled_bs])
            r_outputs_soft = F.softmax(r_outputs, dim=1)
            r_loss_seg_dice = losses.dice_loss(r_outputs_soft[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] == 1)
            if v_loss_seg_dice < r_loss_seg_dice:
                Good_student = 0
            else:
                Good_student = 1

            ###DCR    模块
            v_outputs_soft2 = F.softmax(v_outputs, dim=1)
            r_outputs_soft2 = F.softmax(r_outputs, dim=1)
            v_predict = torch.max(v_outputs_soft2[:labeled_bs, :, :, :, :], 1, )[1]   #troch.max()[1]， 只返回最大值的每个索引
            r_predict = torch.max(r_outputs_soft2[:labeled_bs, :, :, :, :], 1, )[1]
            diff_mask = ((v_predict == 1) ^ (r_predict == 1)).to(torch.int32)# 不同区域的掩码

            ##假设相同预测区域掩码为same_mask   v_outdis
            same_mask = ~diff_mask
            ####  ----------实验(1)开始：对相同区域执行像素级别校正


            v_dis_predict_mask = torch.sigmoid(-1500*v_outdis)
            r_dis_predict_mask = torch.sigmoid(-1500*r_outdis)
            v_dis_predict_mask22 = torch.sigmoid(-1500*v_outdis[:labeled_bs, :, :, :, :])
            r_dis_predict_mask22 = torch.sigmoid(-1500*r_outdis[:labeled_bs, :, :, :, :])
            # 校正损失
            v_mse_dist = consistency_criterion(v_outputs_soft2[:labeled_bs, 1, :, :, :], v_label[:labeled_bs])
            r_mse_dist = consistency_criterion(r_outputs_soft2[:labeled_bs, 1, :, :, :], r_label[:labeled_bs])
            v_mse_dis_dist = torch.mean((v_dis_predict_mask22 - v_outputs_soft[:labeled_bs, :, :, :, :]) ** 2)
            r_mse_dis_dist = torch.mean((r_dis_predict_mask22 - r_outputs_soft[:labeled_bs, :, :, :, :]) ** 2)  
           
            v_mse_dis = torch.sum(same_mask*v_mse_dis_dist)/(torch.sum(same_mask)+1e-16)
            r_mse_dis = torch.sum(same_mask * r_mse_dis_dist) / (torch.sum(same_mask) + 1e-16)
            v_mse = torch.sum(diff_mask * v_mse_dist) / (torch.sum(diff_mask) + 1e-16)
            r_mse = torch.sum(diff_mask * r_mse_dist) / (torch.sum(diff_mask) + 1e-16)


            #距离变换图
            v_dis_to_mask = torch.sigmoid(-1500 * v_outdis)
            v_loss_dis_dice = losses.dice_loss(v_dis_to_mask[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] == 1)
            r_dis_to_mask = torch.sigmoid(-1500 * r_outdis)
            r_loss_dis_dice = losses.dice_loss(r_dis_to_mask[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] == 1)

            ###监督损失
            # v_supervised_loss = (v_loss_seg + v_loss_seg_dice+v_loss_dis_dice) + 0.5* v_mse+0.5*v_mse_dis
            # r_supervised_loss = (r_loss_seg + r_loss_seg_dice+r_loss_dis_dice) + 0.5* r_mse+0.5*r_mse_dis
 
            v_supervised_loss = (v_loss_seg + v_loss_seg_dice+v_loss_dis_dice) + 0.5*v_mse
            r_supervised_loss = (r_loss_seg + r_loss_seg_dice+r_loss_dis_dice) + 0.5*r_mse
           
            # 动态伪标签生成算法
            v_outputs_clone = v_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
            r_outputs_clone = r_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
            v_outputs_clone1 = torch.pow(v_outputs_clone, 1 / T)
            r_outputs_clone1 = torch.pow(r_outputs_clone, 1 / T)
            v_outputs_clone2 = torch.sum(v_outputs_clone1, dim=1, keepdim=True)
            r_outputs_clone2 = torch.sum(r_outputs_clone1, dim=1, keepdim=True)
            v_outputs_PLable = torch.div(v_outputs_clone1, v_outputs_clone2)
            r_outputs_PLable = torch.div(r_outputs_clone1, r_outputs_clone2)
            # 选择伪标签生成器
            if Good_student == 0:
                Plabel = v_outputs_PLable
            if Good_student == 1:
                Plabel = r_outputs_PLable
            # 权重更新策略
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # 计算一致性损失，利用 无标签数据
            if Good_student == 0:
                _,outputs_r = model_resnet(inputs_r_noise)  # 无标记数据+噪声过A网
                outputs_r = transforms_back_rot(outputs_r, r_rot_mask, r_flip_mask)  # B网输出后进行旋转变化
                r_consistency_dist = 0.35*losses.softmax_mse_loss_three(outputs_r, r_outputs_soft[labeled_bs:, :, :, :, :],v_outputs_PLable)
                #r_consistency_dist = 0.5*losses.softmax_mse_loss_three(outputs_r, r_outputs_soft[labeled_bs:, :, :, :, :],v_outputs_PLable)
                #r_consistency_dist = consistency_criterion  ( r_outputs_soft[labeled_bs:, :, :, :, :],v_outputs_PLable)
                b, c, w, h, d = r_consistency_dist.shape
                r_consistency_dist = torch.sum(r_consistency_dist) / (b * c * w * h * d)
                r_consistency_loss = r_consistency_dist
                v_cross_task_dist = torch.mean((v_dis_to_mask - v_outputs_soft) ** 2)
                ##回归任务损失
                r_loss_dis_dice = losses.dice_loss(r_dis_to_mask[labeled_bs:, 1, :, :, :], r_label[labeled_bs:] == 1)
                # r网络上的  双任务损失
                r_cross_task_dist = torch.mean((r_dis_to_mask - r_outputs_soft) ** 2)
                v_loss = v_supervised_loss+v_cross_task_dist*consistency_weight
                r_loss = r_supervised_loss+consistency_weight * (r_loss_dis_dice+r_consistency_loss+r_cross_task_dist)
                writer.add_scalar('loss/r_consistency_loss', r_consistency_loss, iter_num)
            if Good_student == 1:
                _,outputs_v = model_resnet(inputs_v_noise)  # 无标记数据+噪声过A网
                outputs_v = transforms_back_rot(outputs_v, v_rot_mask, v_flip_mask)  # B网输出后进行旋转变化
                v_consistency_dist = 0.35*losses.softmax_mse_loss_three(outputs_v, v_outputs_soft[labeled_bs:, :, :, :, :],r_outputs_PLable)
                #v_consistency_dist = consistency_criterion( v_outputs_soft[labeled_bs:, :, :, :, :],r_outputs_PLable)
                b, c, w, h, d = v_consistency_dist.shape
                v_consistency_dist = torch.sum(v_consistency_dist) / (b * c * w * h * d)
                v_consistency_loss = v_consistency_dist
                v_loss_dis_dice = losses.dice_loss(v_dis_to_mask[labeled_bs:, 1, :, :, :], v_label[labeled_bs:] == 1)
                v_cross_task_dist = torch.mean((v_dis_to_mask - v_outputs_soft) ** 2)
                # r网络上的  双任务损失
                r_cross_task_dist = torch.mean((r_dis_to_mask - r_outputs_soft) ** 2)
                v_loss = v_supervised_loss + consistency_weight * (v_loss_dis_dice+v_consistency_loss+v_cross_task_dist)
                r_loss = r_supervised_loss+r_cross_task_dist*consistency_weight
                writer.add_scalar('loss/v_consistency_loss', v_consistency_loss, iter_num)
            vnet_optimizer.zero_grad()
            resnet_optimizer.zero_grad()
            v_loss.backward()
            r_loss.backward()
            vnet_optimizer.step()
            resnet_optimizer.step()
            logging.info(
                'iteration ： %d v_supervised_loss : %f v_loss_seg : %f v_loss_seg_dice : %f v_loss_mse : %f r_supervised_loss : %f r_loss_seg : %f r_loss_seg_dice : %f r_loss_mse : %f Good_student: %f' %
                (iter_num,
                 v_supervised_loss.item(), v_loss_seg.item(), v_loss_seg_dice.item(), v_mse.item(),
                 r_supervised_loss.item(), r_loss_seg.item(), r_loss_seg_dice.item(), r_mse.item(), Good_student))

            ## change lr
            if iter_num % 2500 == 0 and iter_num != 0:
                lr_ = lr_ * 0.1
                for param_group in vnet_optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in resnet_optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0 and iter_num != 0:
                save_mode_path_vnet = os.path.join(snapshot_path, 'train4_vnet_iter_' + str(iter_num) + '.pth')
                torch.save(model_vnet.state_dict(), save_mode_path_vnet)
                save_mode_path_resnet = os.path.join(snapshot_path, 'train4_resnet_iter_' + str(iter_num) + '.pth')
                torch.save(model_resnet.state_dict(), save_mode_path_resnet)
            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
    save_mode_path_vnet = os.path.join(snapshot_path, 'train4_vnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_vnet.state_dict(), save_mode_path_vnet)
    logging.info("save model to {}".format(save_mode_path_vnet))
    save_mode_path_resnet = os.path.join(snapshot_path, 'train4_resnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_resnet.state_dict(), save_mode_path_resnet)
    logging.info("save model to {}".format(save_mode_path_resnet))
    writer.close()


