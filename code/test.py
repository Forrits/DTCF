import os
import argparse
import torch
from networks.vnet_sdf import VNet
from networks.ResNet34_sdf import Resnet34
from test_util import LA_test_all_case
from networks.urpy import unet_3D_dv_semi
# from networks.vnet_sdf import VNet


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='dataset', help='Name of Experiment')  # todo change dataset path
parser.add_argument('--model', type=str,  default="MCF_flod0", help='model_name')                # todo change test model name
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "./2024.9.21./model/"+FLAGS.model+'urpc/la/'

test_save_path = "./2024.9.21./prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/LA/Flods/test4.list', 'r') as f:                                         # todo change test flod
    image_list = f.readlines()
image_list = [item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]
print(image_list)
#DTCF的测试
# def create_model(name='vnet'):
#     # Network definition
#     if name == 'vnet':
#         net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
#         model = net.cuda()
#     if name == 'resnet34':
#         net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
#         model = net.cuda()

#     return model

# def create_model(ema=False):
#         # Network definition
#         net = VNet(n_channels=1, n_classes=num_classes-1,
#                    normalization='batchnorm', has_dropout=True)
#         model = net.cuda()
#         if ema:
#             for param in model.parameters():
#                 param.detach_()
#         return model

def create_model():
    model = unet_3D_dv_semi(n_classes=num_classes, in_channels=1).cuda()
    return model
def test_calculate_metric(epoch_num):
    # vnet   = create_model(name='vnet')
    # resnet = create_model(name='resnet34')
    vnet   = create_model()
    v_save_mode_path = os.path.join(snapshot_path, 'train4_vnet_iter_' + str(epoch_num) + '.pth')
   
    vnet.load_state_dict(torch.load(v_save_mode_path))
    
    vnet.eval() 

   
   # r_save_mode_path = os.path.join(snapshot_path, 'train4_resnet_iter_' + str(epoch_num) + '.pth')
   # resnet.load_state_dict(torch.load(r_save_mode_path))
    # print("init weight from {}".format(r_save_mode_path))
   # resnet.eval()

    # print("init weight from {}".format(r_save_mode_path))
   

    avg_metric = LA_test_all_case(vnet,None, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric

if __name__ == '__main__':
    iters = 6000
    metric = test_calculate_metric(iters)
    print('iter:', iter)
    print(metric)
