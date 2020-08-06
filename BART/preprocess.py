from tqdm import tqdm
import torch
from transformers import BartTokenizer
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os

from others.logging import logger, init_logger
from others.utils import pad_sents, save, get_max_len, get_mask


class MultiNewsDataset(Dataset):
    def __init__(self, tokenizer, multi_news_reader, mode, args):
        self.tokenizer = tokenizer
        self.multi_news_reader = multi_news_reader
        self.mode = mode
        if mode == 'train':
            self.src = multi_news_reader.train_data_src
            self.tgt = multi_news_reader.train_data_tgt
        self.src = [i.strip('\n') for i in self.src]
        self.tgt = [i.strip('\n') for i in self.tgt]
        if args.minor_data:
            index=list(range(0,len(self.src)))
            chosen_idex = random.sample(index, int(len(self.src)*args.percentage/100))
            self.src = [self.src[i] for i in chosen_idex]
            self.tgt = [self.tgt[i] for i in chosen_idex]
        self.src_ids = self.tokenize(self.src)
        self.tgt_ids = self.tokenize(self.tgt)
        self.src_ids = [i[1:-1] for i in self.src_ids]
        self.tgt_ids = [i[1:-1] for i in self.tgt_ids]
        src_max_len = get_max_len(self.src_ids)
        tgt_max_len = get_max_len(self.tgt_ids)
        self.max_len = max(src_max_len, tgt_max_len)
        logger.info('max_len = {}'.format(self.max_len))

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx]

    def tokenize(self, data):
        # tokenized_text = [self.tokenizer.encode(i,max_length=600) for i in tqdm(data)]
        tokenized_text = [self.tokenizer.encode(i) for i in tqdm(data)]
        return tokenized_text

    def collate_fn(self, data):
        raw_src = [pair[0] for pair in data]
        raw_tgt = [pair[1] for pair in data]
        src = []
        tgt = []
        for i in range(len(raw_src)):
            if (raw_src[i] != []) and (raw_tgt[i] != []):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
        mask = torch.tensor(get_mask(src, max_len=1024))
        src_ids = torch.tensor(pad_sents(src, 1, max_len=1024)[0])
        decoder_ids = [[0]+i for i in tgt]
        label_ids = [i+[2] for i in tgt]

        decoder_ids = torch.tensor(pad_sents(decoder_ids, 1, max_len=256)[0])
        label_ids = torch.tensor(pad_sents(label_ids, -100, max_len=256)[0])
        return src_ids, decoder_ids, mask, label_ids

class MultiNewsReader(object):
    '''
    Load MultiNews Data, Preprocess data, return pytorch dataloaders (Train_loader, Val_loader, Test_loader)
    Required args.data_path
    '''
    def __init__ (self, args):
        self.args = args
        self.raw_data = self.load_multinews_data(self.args)
        self.train_data_src = self.raw_data[0]
        self.train_data_tgt = self.raw_data[1]

    def file_reader(self, file_path):
        file = open(file_path, 'r')
        lines = file.readlines()
        return lines

    def load_multinews_data(self, args):
        train_src_path = args.data_path + args.data_name + '/' + args.mode + '.source'
        train_tgt_path = args.data_path + args.data_name + '/' + args.mode + '.target'


        train_src_lines = self.file_reader(train_src_path)
        train_tgt_lines = self.file_reader(train_tgt_path)
        return (train_src_lines, train_tgt_lines)



def multi_news_builder(args):
    save_path = args.data_path + args.data_name + '/' + args.mode + 'loader' + str(args.percentage) + '.pt'
    if args.minor_data:
        save_path = args.data_path + args.data_name + '/minor_data' + '/' + args.mode + 'loader' + str(args.percentage) + '.pt'
    multi_news_reader = MultiNewsReader(args)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    train_set = MultiNewsDataset(tokenizer, multi_news_reader, 'train', args)
    if args.mode == 'train':
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
    else:
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False, collate_fn=train_set.collate_fn)
    print(train_set.max_len)
    save(train_loader, save_path)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default='../datasets/', type=str)
    parser.add_argument('-data_name', default='debate', type=str)
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-batch_size', default=4, type=int)
    parser.add_argument('-random_seed', type=int, default=199744)
    parser.add_argument('-minor_data', action='store_true')
    parser.add_argument('-percentage', default=100, type=int)
    args = parser.parse_args()

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    # if use minor data
    if args.minor_data:
        print('makeing dataset for {}% data'.format(str(args.percentage)))
        if not os.path.exists(args.data_path + args.data_name + '/minor_data'):
            os.makedirs(args.data_path + args.data_name + 'minor_data')
    multi_news_builder(args)

