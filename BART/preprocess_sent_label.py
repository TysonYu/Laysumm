import numpy as np
import copy
from nltk.tokenize import sent_tokenize
from cal_rouge import test_rouge, rouge_results_to_str

# def calrouge(summary, reference):
#     final_results = test_rouge([summary], [reference], 1)
#     R1_F1 = final_results["rouge_1_f_score"] * 100
#     R2_F1 = final_results["rouge_2_f_score"] * 100
#     Rl_F1 = final_results["rouge_l_f_score"] * 100
#     return (R1_F1+R2_F1+Rl_F1)/3

# def rouge_eval(summary, reference):
#     final_results = test_rouge([summary], [reference], 1)
#     R1_F1 = final_results["rouge_1_f_score"] * 100
#     R2_F1 = final_results["rouge_2_f_score"] * 100
#     Rl_F1 = final_results["rouge_l_f_score"] * 100
#     return (R1_F1+R2_F1+Rl_F1)/3

def calrouge(summary, reference, rouge):
    rouge.add(summary, reference)
    final_results = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL" ])
    R1_F1 = final_results["rouge1"].mid.fmeasure * 100
    R2_F1 = final_results["rouge2"].mid.fmeasure * 100
    Rl_F1 = final_results["rougeL"].mid.fmeasure * 100
    return (R1_F1+R2_F1+Rl_F1)/3

def rouge_eval(summary, reference, rouge):
    rouge.add(summary, reference)
    final_results = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL" ])
    R1_F1 = final_results["rouge1"].mid.fmeasure * 100
    R2_F1 = final_results["rouge2"].mid.fmeasure * 100
    Rl_F1 = final_results["rougeL"].mid.fmeasure * 100
    return (R1_F1+R2_F1+Rl_F1)/3

article = sent_tokenize(' The burden of hepatitis E virus (HEV) infection among patients with haematological malignancy has only been scarcely reported. Therefore, we aimed to describe this burden in patients with haematological malignancies, including those receiving allogeneic haematopoietic stem cell transplantation. We conducted a retrospective, multicentre cohort study across 11 European centres and collected clinical characteristics of 50 patients with haematological malignancy and RNA-positive, clinically overt hepatitis E between April 2014 and March 2017. The primary endpoint was HEV-associated mortality; the secondary endpoint was HEV-associated liver-related morbidity. The most frequent underlying haematological malignancies were aggressive non-Hodgkin lymphoma (NHL) (34%), indolent NHL (iNHL) (24%), and acute leukaemia (36%). Twenty-one (42%) patients had received allogeneic haematopoietic stem cell transplantation (alloHSCT). Death with ongoing hepatitis E occurred in 8 (16%) patients, including 1 patient with iNHL and 1 patient >100 days after alloHSCT in complete remission, and was associated with male sex (p = 0.040), cirrhosis (p = 0.006) and alloHSCT (p = 0.056). Blood-borne transmission of hepatitis E was demonstrated in 5 (10%) patients, and associated with liver-related mortality in 2 patients. Hepatitis E progressed to chronic hepatitis in 17 (34%) patients overall, and in 10 (47.6%) and 6 (50%) alloHSCT and iNHL patients, respectively. Hepatitis E was associated with acute or acute-on-chronic liver failure in 4 (8%) patients with 75% mortality. Ribavirin was administered to 24 (48%) patients, with an HEV clearance rate of 79.2%. Ribavirin treatment was associated with lower mortality (p = 0.037) and by trend with lower rates of chronicity (p = 0.407) when initiated <24 and <12 weeks after diagnosis of hepatitis E, respectively. Immunosuppressive treatment reductions were associated with mortality in 2 patients (28.6%). Hepatitis E is associated with mortality and liver-related morbidity in patients with haematological malignancy. Blood-borne transmission contributes to the burden. Ribavirin should be initiated early, whereas reduction of immunosuppressive treatment requires caution.')


abstract = 'Little is known about the burden of hepatitis E among patients with haematological malignancy. We conducted a retrospective European cohort study among 50 patients with haematological malignancy, including haematopoietic stem cell transplant recipients, with clinically significant HEV infection and found that hepatitis E is associated with hepatic and extrahepatic mortality, including among patients with indolent disease or among stem cell transplant recipients in complete remission. Hepatitis E virus infection evolved to chronic hepatitis in 5 (45.5%) patients exposed to a rituximab-containing regimen and 10 (47.6%) stem cell transplant recipients. Reducing immunosuppressive therapy because of hepatitis E was associated with mortality'

def calLabel(article, abstract, rouge):
    hyps_list = article
    refer = abstract
    scores = []
    for hyps in hyps_list:
        mean_score = calrouge(hyps, refer, rouge)
        scores.append(mean_score)

    selected = [int(np.argmax(scores))]
    selected_sent_cnt = 1

    best_rouge = np.max(scores)
    while selected_sent_cnt < len(hyps_list):
        cur_max_rouge = 0.0
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.append(i)
                hyps = "\n".join([hyps_list[idx] for idx in np.sort(temp)])
                cur_rouge = rouge_eval(hyps, refer, rouge)
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_idx = i
        if cur_max_rouge != 0.0 and cur_max_rouge >= best_rouge:
            selected.append(cur_max_idx)
            selected_sent_cnt += 1
            best_rouge = cur_max_rouge
        else:
            break
    # print(selected, best_rouge)
    return selected

# select = calLabel(article, abstract)
# select =[3, 2, 4, 5, 7]
# print(select)

# new = ' '.join([article[i] for i in select])
# # s1 = rouge_test(new, abstract)
# s2 = calrouge(' '.join(article), abstract)

# s3 = calrouge(new, abstract)
# # print(s1)
# print(s2)
# print(s3)
# import nlp
# rouge = nlp.load_metric('rouge')

# select = calLabel(article, abstract, rouge)
# select = [2, 0, 8, 13, 5,3]
# print(select)
# new = ' '.join([article[i] for i in select])
# s1 = rouge_test(new, abstract)
# s2 = calrouge(new, abstract, rouge)
# print(s1)
# print(s2)

# ==============================================================================================================================================
from tqdm import tqdm
import torch
from transformers import BartTokenizer
from nltk.tokenize import sent_tokenize
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
        
        self.src_ids = []
        self.tgt_ids = []
        self.ext_label = []
        self.sent_ids = []
        for i in tqdm(range(len(self.src))):
            src_sent = sent_tokenize(self.src[i])
            golden_list = []
            golden_list = calLabel(src_sent, self.tgt[i], rouge)
            
            sent_label = []
            source_ids = []
            target_ids = self.tokenizer.encode(self.tgt[i], add_special_tokens=False)
            for j in range(len(src_sent)):
                one_ids = self.tokenizer.encode('<s> ' + src_sent[j], add_special_tokens=False)
                sent_label.append(len(source_ids))
                source_ids += one_ids
                
            new_golden_list = []
            for j in range(len(sent_label)):
                if j in golden_list:
                    new_golden_list.append(1)
                else:
                    new_golden_list.append(0)
            self.sent_ids.append(sent_label)
            self.tgt_ids.append(target_ids)
            self.src_ids.append(source_ids)
            self.ext_label.append(new_golden_list)
#         print(self.src_ids[0])
#         print(self.tgt_ids[0])
#         print(self.ext_label[0])
#         print(self.sent_ids[0])
#         exit()
    
    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx], self.ext_label[idx], self.sent_ids[idx]

    def tokenize(self, data):
        # tokenized_text = [self.tokenizer.encode(i,max_length=600) for i in tqdm(data)]
        tokenized_text = [self.tokenizer.encode(i) for i in tqdm(data)]
        return tokenized_text

    def collate_fn(self, data):
        raw_src = [pair[0] for pair in data]
        raw_tgt = [pair[1] for pair in data]
        ext_label = [pair[2] for pair in data]
        sent_ids = [pair[3] for pair in data]
        src = []
        tgt = []
        new_sent_ids = []
        new_ext_label = []
        for i in range(len(sent_ids)):
            line = []
            for ids in sent_ids[i]:
                if ids < 1024:
                    line.append(ids)
            new_sent_ids.append(line)
            new_ext_label.append(ext_label[i][:len(line)])
        
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
        return src_ids, decoder_ids, mask, label_ids, torch.tensor(new_ext_label), torch.tensor(new_sent_ids)

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
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    train_set = MultiNewsDataset(tokenizer, multi_news_reader, 'train', args)
    if args.mode == 'train':
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
    else:
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False, collate_fn=train_set.collate_fn)
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

    import nlp
    rouge = nlp.load_metric('rouge') 
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




