import torch
from tqdm import tqdm
import os
from transformers import BartTokenizer
from others.logging import logger
from cal_rouge import test_rouge, rouge_results_to_str

def train(model, training_data, validation_data, optimizer, checkpoint, args):
    ''' Start training '''
    logger.info('Start training')
    iteration = 0
    if args.train_from!= '':
        iteration = checkpoint['iteration']
    total_loss = 0
    F1 = 0
    for epoch_i in range(args.epoch):
        logger.info('[ Epoch : {}]'.format(epoch_i))

        # training part
        model.train()
        for src_ids, decoder_ids, mask, label_ids in training_data:
            iteration += 1
            src_ids = src_ids.cuda()
            decoder_ids = decoder_ids.cuda()
            mask = mask.cuda()
            label_ids = label_ids.cuda()
            # forward
            # optimizer.optimizer.zero_grad()
            logits = model(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)
            masked_lm_loss = logits[0]
            loss = masked_lm_loss
#             decoder_outputs = logits[1]
#             encoder_outputs = logits[2]
#             print(encoder_outputs)
#             print(encoder_outputs.shape)
#             print(decoder_outputs.shape)
#             exit()
            total_loss += loss.item()
            loss = loss / args.accumulation_steps
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # loss accumulation
            if (iteration+1) % args.accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
            # write to log file
            if iteration % 20 == 0:
                logger.info("iteration: {} loss_per_word: {:4f} learning rate: {:4f}".format(iteration, total_loss/50, optimizer.learning_rate))
                total_loss = 0
            # save model
            if iteration % 2000 == 0 and iteration > 2000:
                temp_F1 = evaluation(model, validation_data, args)
                if temp_F1 > F1:
                    logger.info("saving model")
                    if not os.path.exists(args.saving_path + args.data_name):
                        os.makedirs(args.saving_path + args.data_name)
                    model_name = make_file_name(args, iteration)
                    checkpoint = {'iteration': iteration, 'settings': args, 'optim': optimizer.optimizer.state_dict(), 'model': model.state_dict()}
                    torch.save(checkpoint, model_name)
                    F1 = temp_F1
                else:
                    pass

def evaluation(model, validation_data, args):
    # model.eval()
    valid_reference_path = '../datasets/' + args.data_name + '/valid.target'
    valid_data = open(valid_reference_path,'r')
    valid_list = valid_data.readlines()
    valid_list = [i.strip('\n') for i in valid_list]
    # inference
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    outputs = []
    for src_ids, decoder_ids, mask, label_ids in tqdm(validation_data):
        src_ids = src_ids.cuda()
        summary_ids = model.generate(src_ids, num_beams=4, max_length=150, early_stopping=True)
        output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        outputs += output
    # calculate rouge
    final_results = test_rouge(outputs, valid_list, args.process_num)
    R1_F1 = final_results["rouge_1_f_score"] * 100
    logger.info('[ Validation ]')
    logger.info(rouge_results_to_str(final_results))
    return R1_F1

def make_file_name(args, iteration):
    model_name = args.saving_path + "{}/{}_{}%.chkpt".format(args.data_name,iteration,args.percentage)
    if args.pre_trained_lm != '':
        model_name = args.saving_path + "{}/{}_{}%_pre_trained_lm.chkpt".format(args.data_name,iteration,args.percentage)
    if args.pre_trained_src:
        model_name = args.saving_path + "{}/{}_{}%_pre_trained_src.chkpt".format(args.data_name,iteration,args.percentage)
    return model_name

def train_multi(model, training_data, validation_data, optimizer, checkpoint, args):
    ''' Start training '''
    logger.info('Start training')
    iteration = 0
    if args.train_from!= '':
        iteration = checkpoint['iteration']
    total_loss = 0
    F1 = 0
    for epoch_i in range(args.epoch):
        logger.info('[ Epoch : {}]'.format(epoch_i))

        # training part
        model.train()
        for src_ids, decoder_ids, mask, label_ids, ext_label, sent_ids in training_data:
            iteration += 1
            src_ids = src_ids.cuda()
            decoder_ids = decoder_ids.cuda()
            mask = mask.cuda()
            label_ids = label_ids.cuda()
            ext_label = ext_label.cuda()
            sent_ids = sent_ids.cuda()
            # forward
            # optimizer.optimizer.zero_grad()
            loss = model(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids, ext_label=ext_label, sent_ids=sent_ids)[0]
            total_loss += loss.item()
            loss = loss / args.accumulation_steps
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # loss accumulation
            if (iteration+1) % args.accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
            # write to log file
            if iteration % 20 == 0:
                logger.info("iteration: {} loss_per_word: {:4f} learning rate: {:4f}".format(iteration, total_loss/50, optimizer.learning_rate))
                total_loss = 0
            # save model
            if iteration % 100 == 0 and iteration > 2000:
                temp_F1 = evaluation_multi(model, validation_data, args)
                if temp_F1 > F1:
                    logger.info("saving model")
                    if not os.path.exists(args.saving_path + args.data_name):
                        os.makedirs(args.saving_path + args.data_name)
                    model_name = make_file_name(args, iteration)
                    checkpoint = {'iteration': iteration, 'settings': args, 'optim': optimizer.optimizer.state_dict(), 'model': model.state_dict()}
                    torch.save(checkpoint, model_name)
                    F1 = temp_F1
                else:
                    pass
                
def evaluation_multi(model, validation_data, args):
    # model.eval()
    valid_reference_path = '../datasets/' + args.data_name + '/valid.target'
    valid_data = open(valid_reference_path,'r')
    valid_list = valid_data.readlines()
    valid_list = [i.strip('\n') for i in valid_list]
    # inference
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    outputs = []
    for src_ids, decoder_ids, mask, label_ids, ext_label, sent_ids in tqdm(validation_data):
        src_ids = src_ids.cuda()
        summary_ids = model.generate(src_ids, num_beams=4, max_length=150, early_stopping=True)
        output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        outputs += output
    # calculate rouge
    final_results = test_rouge(outputs, valid_list, args.process_num)
    R1_F1 = final_results["rouge_1_f_score"] * 100
    logger.info('[ Validation ]')
    logger.info(rouge_results_to_str(final_results))
    return R1_F1