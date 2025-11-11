import os
import argparse
import torch
import dgl
import numpy as np
from utils import  construct_graph, split_flow_Tor_nonoverlapping, split_flow_ISCX
from config import *

def transform_data(data,l):
    res = []
    for i,file in enumerate(data):
        tmp = []
        res_tmp = []
        length = len(file)
        for key in file:
            bit = bin(key)
            bit = bit[2:]
            tmp.extend(bit)
        for i,_ in enumerate(tmp):
            bit = tmp[i*l:i*l+l]
            bit = ''.join(bit)
            bit = bit.zfill(8)
            res_tmp.append(bit)
        tmp = [int(hex, 2) for hex in res_tmp]
        tmp = tmp[:length]
        res.append(tmp)
        tmp = []
    return res

def train2bit(train_path,test_path,type,transform_length,save_train_path,save_test_path):
    train = np.load(train_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    if type == 'payload':
        data = train['data'].reshape(-1, config.BYTE_PAD_TRUNC_LENGTH)
        test_data = test['data'].reshape(-1, config.BYTE_PAD_TRUNC_LENGTH)
    elif type == 'header':
        data = train['data'].reshape(-1, config.HEADER_BYTE_PAD_TRUNC_LENGTH)
        test_data = test['data'].reshape(-1, config.HEADER_BYTE_PAD_TRUNC_LENGTH)
    print(data.shape,test_data.shape)
    label = train['label']
    test_label = test['label']
    data = transform_data(data=data,l=transform_length)
    test_data = transform_data(data=test_data,l=transform_length)
    print(np.array(data).shape,np.array(test_data).shape)
    if type == 'payload':
        np.savez_compressed(save_train_path, data=np.array(data), label=np.array(label))
        np.savez_compressed(save_test_path, data=np.array(test_data), label=np.array(test_label))
    elif type == 'header':
        np.savez_compressed(save_train_path, data=np.array(data), label=np.array(label))
        np.savez_compressed(save_test_path, data=np.array(test_data), label=np.array(test_label))

def construct_dataset_from_bytes_ISCX(dir_path_dict, type):
    train = []
    train_label = []
    test = []
    test_label = []
    TRAIN_FLOW_COUNT = dict()
    TEST_FLOW_COUNT = dict()
    for category in dir_path_dict:
        dir_path = dir_path_dict[category]
        file_list = os.listdir(dir_path)
        data_list = []
        for file in file_list:
            if not file.endswith('.npz'):
                continue
            file_path = dir_path + '/' + file
            print('{} Process Starting'.format(file_path))
            if opt.dataset == 'iscx-tor':
                data_list.extend(split_flow_Tor_nonoverlapping(file_path, category, allow_empty=False, pad_trunc=True, config=config, type=type))
            else:
                data_list.extend(split_flow_ISCX(file_path, category, allow_empty=False, pad_trunc=True, config=config, type=type))

        data_list = data_list[:config.MAX_SEG_PER_CLASS]
        split_ind = int(len(data_list) / 10)
        data_list_train = data_list[split_ind + 1:]
        data_list_test = data_list[: split_ind + 1]
        
        train.extend(data_list_train)
        train_label.extend([category] * len(data_list_train))
        test.extend(data_list_test)
        test_label.extend([category] * len(data_list_test))

        TRAIN_FLOW_COUNT[category] = len(data_list_train)
        TEST_FLOW_COUNT[category] = len(data_list_test)
        print(TRAIN_FLOW_COUNT[category], TEST_FLOW_COUNT[category])
        print(config.TRAIN_DATA)
        
    if type == 'payload':
        np.savez_compressed(config.TRAIN_DATA, data=np.array(train), label=np.array(train_label))
        np.savez_compressed(config.TEST_DATA, data=np.array(test), label=np.array(test_label))
    elif type == 'header':
        np.savez_compressed(config.HEADER_TRAIN_DATA, data=np.array(train), label=np.array(train_label))
        np.savez_compressed(config.HEADER_TEST_DATA, data=np.array(test), label=np.array(test_label))

    print(TRAIN_FLOW_COUNT)
    print(TEST_FLOW_COUNT)

def construct_graph_format_data(file_path_header,file_path_payload, save_path, w_size, pmi=1):
    file_header = np.load(file_path_header, allow_pickle=True)
    file_payload = np.load(file_path_payload, allow_pickle=True)
    gs = []
    data_payload = file_payload['data'].reshape(-1, config.BYTE_PAD_TRUNC_LENGTH)
    data_header = file_header['data'].reshape(-1, config.HEADER_BYTE_PAD_TRUNC_LENGTH)
    label = file_header['label']
    ind = 0
    for h, p in zip(data_header,data_payload):
        g = construct_graph(bytes=p,header=h, w_size=w_size, k=pmi)
        gs.append(g)
        ind = ind+1
        if ind % 500 == 0:
            print('{} Graphs Constructed'.format(ind))

    dgl.save_graphs(save_path, gs, {"glabel": torch.LongTensor(label)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--transform_length", type=int, help="transform_length", required=True,default=4)
    opt = parser.parse_args()

    if opt.dataset == 'iscx-vpn':
        config = ISCXVPNConfig()
    elif opt.dataset == 'iscx-nonvpn':
        config = ISCXNonVPNConfig()
    elif opt.dataset == 'iscx-tor':
        config = ISCXTorConfig()
    elif opt.dataset == 'iscx-nontor':
        config = ISCXNonTorConfig()
    elif opt.dataset == 'ciciot':
        config = CICIoTConfig()
    else:
        raise Exception('Dataset Error')

    construct_dataset_from_bytes_ISCX(dir_path_dict=config.DIR_PATH_DICT, type='payload')
    construct_dataset_from_bytes_ISCX(dir_path_dict=config.DIR_PATH_DICT, type='header')
    
    train_path_file = config.TRAIN_DATA
    header_train_file = config.HEADER_TRAIN_DATA
    test_path_file = config.TEST_DATA
    header_test_file = config.HEADER_TEST_DATA
    
    if opt.transform_length == 2:
        save_train_path_file = config.TRAIN_DATA_2bit
        save_header_train_file = config.HEADER_TRAIN_DATA_2bit
        save_test_path_file = config.TEST_DATA_2bit
        save_header_test_file = config.HEADER_TEST_DATA_2bit
    elif opt.transform_length == 4:
        save_train_path_file = config.TRAIN_DATA_4bit
        save_header_train_file = config.HEADER_TRAIN_DATA_4bit
        save_test_path_file = config.TEST_DATA_4bit
        save_header_test_file = config.HEADER_TEST_DATA_4bit
    elif opt.transform_length == 10:
        save_train_path_file = config.TRAIN_DATA_10bit
        save_header_train_file = config.HEADER_TRAIN_DATA_10bit
        save_test_path_file = config.TEST_DATA_10bit
        save_header_test_file = config.HEADER_TEST_DATA_10bit
    else:
        raise Exception('Transform Length Error')
    
    train2bit(train_path_file,test_path_file,"payload",opt.transform_length,save_train_path_file,save_test_path_file)
    train2bit(header_train_file,header_test_file,"header",opt.transform_length,save_header_train_file,save_header_test_file)
    
    construct_graph_format_data(file_path_header=config.HEADER_TRAIN_DATA,file_path_payload=config.TRAIN_DATA, save_path=config.TRAIN_GRAPH_COMBINE,w_size=config.PMI_WINDOW_SIZE)
    construct_graph_format_data(file_path_header=config.HEADER_TEST_DATA, file_path_payload=config.TEST_DATA,save_path=config.TEST_GRAPH_COMBINE,w_size=config.PMI_WINDOW_SIZE)
    construct_graph_format_data(file_path_header=save_header_train_file,file_path_payload=save_train_path_file, save_path=config.TRAIN_GRAPH_COMBINE_4bit,w_size=config.PMI_WINDOW_SIZE)
    construct_graph_format_data(file_path_header=save_header_test_file, file_path_payload=save_test_path_file,save_path=config.TEST_GRAPH_COMBINE_4bit,w_size=config.PMI_WINDOW_SIZE)
    
    
    
