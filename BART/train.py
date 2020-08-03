import os
import argparse
import torch

from transformers import BartForConditionalGeneration
from others.logging import init_logger, logger
from others.utils import load, count_parameters, initialize_weights
from preprocess import MultiNewsDataset, MultiNewsReader
from others.optimizer import build_optim
from trainer import train, train_multi
import random
import numpy as np

def make_log_file_name(args):
    if args.pre_trained_lm != '':
        log_file_name = args.log_file + args.data_name + '/train_'  + args.pre_trained_lm.split('/')[-1][:-3] + '_' + args.percentage + '%_pretrain_lm.log'
    elif args.pre_trained_src:
        log_file_name = args.log_file + args.data_name + '/train_' + args.percentage + '_' + '%_pretrain_src.log'
    else:
        log_file_name = args.log_file + args.data_name + '/train_' + args.percentage + '%.log'
    
    return log_file_name

def load_dataloader(args):
    train_file_name = '../datasets/' + args.data_name + '/trainloader100.pt'
    if args.minor_data:
        train_file_name = '../datasets/' + args.data_name + '/minor_data/trainloader{}.pt'.format(str(args.percentage))
    train_loader = load(train_file_name)
    valid_file_name = '../datasets/' + args.data_name + '/validloader100.pt'
    valid_loader = load(valid_file_name)
    logger.info('train loader has {} samples'.format(len(train_loader.dataset)))
    return train_loader, valid_loader

def load_model(args):
    if args.customiza_model:
        from module.multi_task_model import multi_task_model
        model = multi_task_model.from_pretrained('facebook/bart-large-cnn')
        checkpoint = None
        return model, checkpoint
        
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    if args.pre_trained_lm != '':
        model = torch.load(args.pre_trained_lm, map_location='cpu')
    # load from saved model
    if args.train_from != '':
        logger.info("train from : {}".format(args.train_from))
        checkpoint = torch.load(args.train_from, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    if args.train_from == '':
        checkpoint = None
    return model, checkpoint

if __name__ == '__main__':
    # for training
    parser = argparse.ArgumentParser()
    parser.add_argument('-visible_gpu', default='1', type=str)
    parser.add_argument('-log_file', default='./logs/', type=str)
    parser.add_argument('-train_from', default='', type=str)
    parser.add_argument('-random_seed', type=int, default=199744)
    parser.add_argument('-lr', default=0.05, type=float)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-saving_path', default='./save/', type=str)
    parser.add_argument('-data_name', default='debate', type=str)
    parser.add_argument('-minor_data', action='store_true')
    parser.add_argument('-pre_trained_lm', default='', type=str)
    parser.add_argument('-pre_trained_src', action='store_true')
    parser.add_argument('-customiza_model', action='store_true')
    parser.add_argument('-percentage', default='100', type=str)
    # for learning, optimizer
    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.998, type=float)
    parser.add_argument('-warmup_steps', default=1000, type=int)
    parser.add_argument('-decay_method', default='noam', type=str)
    parser.add_argument('-enc_hidden_size', default=768, type=int)
    parser.add_argument('-clip', default=1.0, type=float)
    parser.add_argument('-accumulation_steps', default=10, type=int)
    # for evaluation
    parser.add_argument('-process_num', default=4, type=int)
    args = parser.parse_args()

    # initial logger
    if not os.path.exists(args.log_file+args.data_name):
        os.makedirs(args.log_file+args.data_name+'/')
    init_logger(make_log_file_name(args))
    logger.info(args)
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    # loading data
    # it's faster to load data from pre_build data
    logger.info('starting to read dataloader')
    train_loader, valid_loader = load_dataloader(args)

    # initial model
    logger.info('starting to build model')
    model, checkpoint = load_model(args)
    model.cuda()
    # initial optimizer
    optim = build_optim(args, model, checkpoint)

    # training
    train_multi(model, train_loader, valid_loader, optim, checkpoint, args)

