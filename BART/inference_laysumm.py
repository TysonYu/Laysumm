import os
import argparse
import torch
from tqdm import tqdm
from transformers import BartTokenizer
from transformers import BartForConditionalGeneration
from others.logging import init_logger, logger
from others.utils import load, count_parameters, initialize_weights
from preprocess import MultiNewsDataset, MultiNewsReader
from others.optimizer import build_optim
from trainer import train


def tokenize(data, tokenizer):
    # tokenized_text = [self.tokenizer.encode(i,max_length=600) for i in tqdm(data)]
    tokenized_text = [tokenizer.encode(i) for i in data]
    return tokenized_text
    
if __name__ == '__main__':
    # for training
    parser = argparse.ArgumentParser()
    parser.add_argument('-visible_gpu', default='1', type=str)
    parser.add_argument('-log_file', default='./logs/laysumm_inference', type=str)
    parser.add_argument('-train_from', default='', type=str)
    parser.add_argument('-random_seed', type=int, default=199744)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-saving_path', default='./save/', type=str)
    parser.add_argument('-data_name', default='debate', type=str)
    parser.add_argument('-customiza_model', action='store_true')
    # for learning, optimizer
    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.998, type=float)
    parser.add_argument('-warmup_steps', default=1000, type=int)
    parser.add_argument('-decay_method', default='noam', type=str)
    parser.add_argument('-enc_hidden_size', default=768, type=int)
    parser.add_argument('-clip', default=1.0, type=float)
    parser.add_argument('-accumulation_steps', default=10, type=int)
    
    args = parser.parse_args()
    
    # initial logger
    init_logger(args.log_file+'_'+args.data_name+'.log')
    logger.info(args)
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    # device = torch.device('cuda: {}'.format(args.visible_gpu) if torch.cuda.is_available() else 'cpu')
    # set random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    # initial tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    
    # initial model
    logger.info('starting to build model')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    if args.customiza_model:
        from module.multi_task_model import multi_task_model
        model = multi_task_model.from_pretrained('facebook/bart-large-cnn')
    if args.train_from != '':
        logger.info("train from : {}".format(args.train_from))
        checkpoint = torch.load(args.train_from, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.cuda()
    
    
    # read data and inference
    main_path = '../datasets/test/'
    test_list = os.listdir(main_path)
    test_list = [line[:17] for line in test_list]
    test_list = list(set(test_list))
    for i in range(37):
        id = test_list[i]
        if id[0] == '.':
            continue
        abstract_path = main_path+id+'_ABSTRACT'
        data = open(abstract_path)
        lines = data.readlines()
        lines = [i.strip('\n') for i in lines]
        lines = [i for i in lines if i != '']
        dio = 'https://doi.org/' + lines[0]
        title = lines[3]
        lines = lines[lines.index('PARAGRAPH')+1:]
        lines = [line for line in lines if line!='PARAGRAPH']
        one_abstract = " ".join(lines)
        
        # 处理introduction
        full_path = main_path+id+'_FULLTEXT'
        data = open(full_path)
        lines = data.readlines()
        lines = [i.strip('\n') for i in lines]
        lines = [i for i in lines if i != '']
        try: 
            lines = lines[lines.index('Introduction')+2:]
            one_intro_list = []
            for line in lines:
                if line == 'PARAGRAPH':
                    break
                one_intro_list.append(line)
            one_intro = " ".join(one_intro_list)
        except:
            print(full_path,'no introduction')
            one_intro = ""
        
        # 处理conclusion
        full_path = main_path+id+'_FULLTEXT'
        data = open(full_path)
        lines = data.readlines()
        lines = [i.strip('\n') for i in lines]
        lines = [i for i in lines if i != '']
        try: 
            lines = lines[lines.index('Conclusions')+2:]
            one_conclu_list = []
            for line in lines:
                if line == 'PARAGRAPH':
                    break
                one_conclu_list.append(line)
            one_conclu = " ".join(one_conclu_list)
        except:
            print(full_path,'no conclusion')
            one_conclu = ""
        ARTICLE_TO_SUMMARIZE = one_abstract + ' ' + one_intro #+ ' ' + one_conclu
        if args.customiza_model:
            from nltk.tokenize import sent_tokenize
            ARTICLE_TO_SUMMARIZE_list = sent_tokenize(ARTICLE_TO_SUMMARIZE)
            ARTICLE_TO_SUMMARIZE = " <s> ".join(ARTICLE_TO_SUMMARIZE_list)
            ARTICLE_TO_SUMMARIZE = " <s> " + ARTICLE_TO_SUMMARIZE
        print(ARTICLE_TO_SUMMARIZE)
#         inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=512, return_tensors='pt', add_special_tokens=False, truncation=True)
        inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE,  add_special_tokens=False,  max_length=1024, truncation=True)
#         summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
        summary_ids = model.generate(torch.tensor([inputs]).cuda(), num_beams=4, max_length=200, early_stopping=True)
        one_summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        summary = one_summary[0]
        summery_list = summary.split()
        summery_list = summery_list[:149]
        summary = " ".join(summery_list)
        file = open('./submit/'+id+'_LAYSUMM.TXT', 'w')
        file.write(dio+'\n\n')
        file.write('LAYSUMM\n\n')
        file.write('TITLE\n\n')
        file.write(title+'\n\n')
        file.write('PARAGRAPH\n\n')
        file.write(summary+'\n')
        file.close()
        logger.info(one_summary)
