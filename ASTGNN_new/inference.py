#!/usr/bin/env python
# coding: utf-8
import argparse
import configparser
import os
import sys
import numpy as np
import torch
from time import time, sleep

import torch_npu
import transfer_to_npu

from lib.utils import get_adjacency_matrix, get_adjacency_matrix_2direction, compute_val_loss, predict_and_save_results, \
    load_graphdata_normY_channel1
from model.ASTGNN import make_model
from lib.utils import re_max_min_normalization

locs = ['金湖东路-红莲街', '金源东路-红莲街', '兴贤路-乐民街', '金湖东路-乐民街', '金源东路-乐民街', '金源西路-乐民街',
        '金湖东路-八于街', '金源东路-八于街', '金源西路-八于街', '金湖东路-乐安北一街', '金源东路-乐安北一街',
        '兴贤路-乐安街', '金源东路-乐安街', '金源东路-双文北街', '金湖东路-乐安街']

loc2idx = {k: v for (v, k) in enumerate(locs)}
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PREDICT.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--location', default='金湖东路-乐民街', help='sencer name')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE, flush=True)

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']
learning_rate = float(training_config['learning_rate'])
start_epoch = int(training_config['start_epoch'])
epochs = int(training_config['epochs'])
fine_tune_epochs = int(training_config['fine_tune_epochs'])
# print('total training epoch, fine tune epoch:', epochs, ',', fine_tune_epochs, flush=True)
batch_size = 12
# batch_size = int(training_config['batch_size'])
# print('batch_size:', batch_size, flush=True)
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
direction = int(training_config['direction'])
encoder_input_size = int(training_config['encoder_input_size'])
decoder_input_size = int(training_config['decoder_input_size'])
dropout = float(training_config['dropout'])
kernel_size = int(training_config['kernel_size'])

# 构建数据集
# all_data = read_and_generate_dataset_encoder_decoder(graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour=points_per_hour, save=True)

filename_npz = os.path.join(
    dataset_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '.npz'
num_layers = int(training_config['num_layers'])
d_model = int(training_config['d_model'])
nb_head = int(training_config['nb_head'])
ScaledSAt = bool(int(training_config['ScaledSAt']))  # whether use spatial self attention
SE = bool(int(training_config['SE']))  # whether use spatial embedding
smooth_layer_num = int(training_config['smooth_layer_num'])
aware_temporal_context = bool(int(training_config['aware_temporal_context']))
TE = bool(int(training_config['TE']))
use_LayerNorm = True
residual_connection = True

if direction == 2:
    adj_mx, distance_mx = get_adjacency_matrix_2direction(adj_filename, num_of_vertices, id_filename)
if direction == 1:
    adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
folder_dir = 'MAE_%s_h%dd%dw%d_layer%d_head%d_dm%d_channel%d_dir%d_drop%.2f_%.2e' % (
    model_name, num_of_hours, num_of_days, num_of_weeks, num_layers, nb_head, d_model, encoder_input_size, direction,
    dropout, learning_rate)

if aware_temporal_context:
    folder_dir = folder_dir + 'Tcontext'
if ScaledSAt:
    folder_dir = folder_dir + 'ScaledSAt'
if SE:
    folder_dir = folder_dir + 'SE' + str(smooth_layer_num)
if TE:
    folder_dir = folder_dir + 'TE'

print('folder_dir:', folder_dir, flush=True)
params_path = os.path.join('./experiments', dataset_name, folder_dir)

net = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx, nb_head, num_of_weeks,
                 num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout,
                 aware_temporal_context=aware_temporal_context, ScaledSAt=ScaledSAt, SE=SE, TE=TE,
                 kernel_size=kernel_size, smooth_layer_num=smooth_layer_num, residual_connection=residual_connection,
                 use_LayerNorm=use_LayerNorm)

print('\n', net, flush=True)
# sleep(2)
print('loading model done!')

timeboard = '21:07'
cut = int(timeboard.split(':')[0]) * 12 + int(timeboard.split(':')[1]) // 5 - 1
print(cut)
location = args.location
if location not in locs:
    print('请输入正确查询地址')
    sys.exit(0)
#sys.exit(0)
print("!!!!num_of_hours",num_of_hours)
print("!!!!num_of_days",num_of_days)
print("!!!!num_of_hours",num_of_hours)
print("!!!!num_of_weeks",num_of_weeks)
print("!!!!DEVICE",DEVICE)
print("!!!!batch_size",batch_size)
# all the input has been normalized into range [-1,1] by MaxMin normalization
train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min = load_graphdata_normY_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size, cut=5)


def predict_main(epoch, data_loader, data_target_tensor, _max, _min, type):
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)  # 原始代码
    print('\nload weight from:', params_filename, flush=True)
    net.load_state_dict(torch.load(params_filename))

    # predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type)
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)  # (B, N, T, 1)

            predict_length = labels.shape[2]  # T

            # encode
            encoder_output = net.encode(encoder_inputs)
            input.append(encoder_inputs[:, :, :, 0:1].cpu().numpy())  # (batch, T', 1)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]  # 只取输入的第一个值作为input，之后都用predict出来的值作为input
            decoder_input_list = [decoder_start_inputs]

            # 按着时间步进行预测
            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]
                # print(f'\n\n#########\n{predict_output[0]}\n########\n\n')
                # print(predict_output[0].size())
                # sys.exit(0)

            prediction.append(predict_output.detach().cpu().numpy())
            # if batch_index % 100 == 0:
            #     print('predicting testing set batch %s / %s, time: %.2fs' % (
            #         batch_index + 1, loader_length, time() - start_time))

            # print('test time on whole data:%.2fs' % (time() - start_time))
            input = np.concatenate(input, 0)
            input = re_max_min_normalization(input, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

            prediction = np.concatenate(prediction, 0)  # (batch, N, T', 1)
            prediction = re_max_min_normalization(prediction, _max[0, 0, 0, 0], _min[0, 0, 0, 0])

    res = ';'.join([str(i[0]) for i in prediction[-1][loc2idx[location]]])
    print(location + '未来1h车流预测为:')
    return res


if __name__ == "__main__":
    result = predict_main(100, test_loader, test_target_tensor, _max, _min, 'test')
    print(result)
