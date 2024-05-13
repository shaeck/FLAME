#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
from random import random
from models.test import test_img
from models.Fed import FedAvg
from models.Nets import ResNet18, vgg19_bn, vgg11_bn, get_model
from models.Client import SwarmClient
from models.BadClient import BadClient
from models.Reputation import Reputation

from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import os
import random
import time
import math
matplotlib.use('Agg')


def write_file(filename, accu_list, time_total, time_aggr, args, analyse = False):
    write_info_to_accfile(filename, args)
    f = open(filename, "a")
    f.write("main_task_accuracy=")
    f.write(str(accu_list))
    f.write('\n')
    f.write("time total per epoch=")
    f.write(str(time_total))
    f.write('\n')
    f.write("time over aggregation per round=")
    f.write(str(time_aggr))
    f.write('\n')
    if analyse == True:
        need_length = len(accu_list)//10
        acc = accu_list[-need_length:]
        best_acc = round(max(acc),2)
        f.write('\n')
        f.write('max acc:')
        f.write(str(best_acc))
        f.write('\n')
        f.close()
        return best_acc
    f.close()


def central_dataset_iid(dataset, dataset_size):
    all_idxs = [i for i in range(len(dataset))]
    central_dataset = set(np.random.choice(all_idxs, dataset_size, replace=False))
    return central_dataset

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    test_mkdir('./'+args.save)
    print_exp_details(args)
    
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            '../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            '../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion_mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        dataset_train = datasets.FashionMNIST(
            '../data/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(
            '../data/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
        else:
            dict_users = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
        else:
            dict_users = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'VGG' and args.dataset == 'cifar':
        net_glob = vgg19_bn(args.dataset).to(args.device)
    elif args.model == "resnet" and args.dataset == 'cifar':
        net_glob = ResNet18(args.dataset).to(args.device)
    elif args.model == "resnet" and args.dataset == 'mnist':
        net_glob = ResNet18(args.dataset).to(args.device)
    elif args.model == "resnet" and args.dataset == 'fashion_mnist':
        net_glob = ResNet18(args.dataset).to(args.device)
    elif args.dataset == 'cifar':
        net_glob = get_model('cifar10').to(args.device)
    elif args.model == "rlr_mnist" or args.model == "cnn":
        net_glob = get_model('fmnist').to(args.device)
    elif args.model == "fcn":
        net_glob = get_model('fcn').to(args.device)
    else:
        exit('Error: unrecognized model')
    
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    
    central_dataset = central_dataset_iid(dataset_test, args.server_dataset)
    base_info = get_base_info(args)
    filename = './'+args.save+'/accuracy_file_{}.txt'.format(base_info)
    
    if args.init != 'None':
        param = torch.load(args.init)
        net_glob.load_state_dict(param)
        print("load init model")

        
    val_acc_list, net_list, time_total_list, time_aggregation_list = [0], [], [], []

    clients = list()
    clients = [SwarmClient(args, dataset_train, i) for i in range(args.num_users)]
    badclients_percentage = int((args.num_users / 100) * 60)
    for i in range(badclients_percentage):
        clients[i] =  BadClient(args, dataset_train, i)
    last_aggregator = 0
    metrics = dict()
    data_length_div = len(dict_users)
        
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for _ in range(args.num_users)]
        reputation = Reputation(clients)
    for iter in range(args.epochs):
        start = time.time()
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_updates = []
        m = max(int(args.frac * args.num_users), 1)
        # m = max(int(args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for num_turn, idx in enumerate(idxs_users):
            data_seg_id = idx % data_length_div
            w, loss = clients[num_turn].update(net_glob, dict_users[data_seg_id])
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        
        w_glob = 0

        time_aggregation_start = time.time()

        #choose client to aggregate here
        if args.swarm:
            #reputation model aggregation
            if args.smart:
                w_glob = clients[last_aggregator].aggregate(w_locals)
            
            #random aggregation
            if args.random:
                client_id = np.random.randint(0, (len(clients)-1))
                w_glob = clients[client_id].aggregate(w_locals)
        if args.federated:
            #federated learning
            w_glob = clients[0].aggregate(w_locals)
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        
        end = time.time()
        # show loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        if iter % 1 == 0:
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Main accuracy: {:.2f}".format(acc_test))
            val_acc_list.append(acc_test.item())
            aggregation_time = end-time_aggregation_start
            time_aggregation_list.append(aggregation_time)
            time_total_list.append(end-start)

            if iter % 20 == 0:
                write_file(filename, val_acc_list, time_total_list, time_aggregation_list, args)

            #choose new aggregator here
            if args.smart:
                # metrics["quality"] = 100*(val_loss_pre - loss_test)
                metrics["quality"] = 100*(val_acc_list[-1] - val_acc_list[-2])
                last_aggregator = reputation.choose_new_aggregator(metrics, last_aggregator)
                val_loss_pre = loss_test
    
    best_acc = write_file(filename, val_acc_list, time_total_list, time_aggregation_list, args, True)
    reputation.print_reputation()
    
    # plot loss curve
    plt.figure()
    plt.xlabel('communication')
    plt.ylabel('accu_rate')
    plt.plot(val_acc_list, label = 'main task(acc:'+str(best_acc)+'%)')
    plt.legend()
    title = base_info
    # plt.title(title, y=-0.3)
    plt.title(title)
    plt.savefig('./'+args.save +'/'+ title + '.pdf', format = 'pdf',bbox_inches='tight')
    
    
    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    