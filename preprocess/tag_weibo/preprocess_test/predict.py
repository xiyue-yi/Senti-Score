import logging
import numpy as np
import argparse
import torch
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import random
from model import BaseModel
from train import Trainer
from dataloader import BertDataset, BertDataLoader
from pytorch_pretrained_bert.modeling import BertModel, BertForSequenceClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

'''
parser = argparse.ArgumentParser(description='nlp work 3')

parser.add_argument('--input-train', type=str, default="preprocessed_data/train.json")
parser.add_argument('--input-dev', type=str, default="preprocessed_data/dev.json")
parser.add_argument('--input-test', type=str, default="preprocessed_data/2019-09-03-2.json")

parser.add_argument('--bert-model', type=str, default="bert-base-chinese",
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                         "bert-base-multilingual-cased, bert-base-chinese."
                    )
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float)
parser.add_argument('--warmup-proportion', type=float, default=0.1)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--dropout-prob', type=float, default=0.1)
parser.add_argument('--weight', type=float, default=1.0)
parser.add_argument('--max-len', type=int, default=80)

parser.add_argument('--save-dir', type=str, default='saved_model/1')
parser.add_argument('--test-result',type=str, default='result/鹿依')
parser.add_argument('--input-dir',type=str, default='preprocessed_data/鹿依')
parser.add_argument('--use-cpu', type=bool, default=True)
parser.add_argument('--gpu-devices', type=str, default="0")
parser.add_argument('--seed', type=int, default=42)

parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--load-model", type=str, default="saved_model/1/ckpt-epoch-10")

args = parser.parse_args()

'''
def predict(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    logging.basicConfig(level=logging.INFO)

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print(PYTORCH_PRETRAINED_BERT_CACHE)
    logging.info("Initializing model...")
    # model = BaseModel(args, use_gpu)
    # model = BertForSequenceClassification.from_pretrained(args.bert_model,
                # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1),
                # num_labels=2)

    model = BertForSequenceClassification.from_pretrained(
                './bert-base-chinese',
                num_labels=2)

    if args.resume:
        model.load_state_dict(torch.load(args.load_model,map_location='cpu'))

    if use_gpu:
        model = model.cuda()

    params = sum(np.prod(p.size()) for p in model.parameters())
    logging.info("Number of parameters: {}".format(params))

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    train_dataset = BertDataset(args.input_train, "train")
    dev_dataset = BertDataset(args.input_dev, "dev")
    #test_dataset = BertDataset(args.input_test, "test")

    # 构建bertdataset
    test_dataset_list = []
    for file in os.listdir(args.input_dir):
        #print(file)
        if '.json' not in file:
            continue
        print(file)
        test_dataset = BertDataset(os.path.join(args.input_dir,file), "test")
        file_name = file[:-5]
        test_dataset_list.append([test_dataset,file_name])
    

    train_examples = len(train_dataset)

    train_dataloader = \
        BertDataLoader(train_dataset, mode="train", max_len=args.max_len, batch_size=args.batch_size, num_workers=4, shuffle=True)
    dev_dataloader = \
        BertDataLoader(dev_dataset, mode="dev", max_len=args.max_len, batch_size=args.batch_size, num_workers=4, shuffle=False)
    #test_dataloader = \
        #BertDataLoader(test_dataset, mode="test", max_len=args.max_len, batch_size=int(args.batch_size / 2), num_workers=4, shuffle=False)
    
    # 构建bertdataloader
    test_dataloader_list = []
    for test_dataset,file_name in test_dataset_list:
        test_dataloader = \
            BertDataLoader(test_dataset, mode="test", max_len=args.max_len, batch_size=int(args.batch_size / 2), num_workers=4, shuffle=False)
        test_dataloader_list.append([test_dataloader,file_name])
    

    trainer = Trainer(args, model, train_examples, use_gpu)

    # 训练阶段
    if args.resume == False:
        logging.info("Beginning training...")
        trainer.train(train_dataloader, dev_dataloader)

    '''
    # 测试阶段
    prediction, id = trainer.predict(test_dataloader)
    result_name = os.path.join(args.test_result,'result_0.txt')
    with open(result_name, "w", encoding="utf-8") as f:
            for index in range(len(prediction)):
                f.write("{}\t{}\n".format(id[index], prediction[index]))
    '''
    if not os.path.exists(args.test_result):
        os.mkdir(args.test_result)

    for test_dataloader,file_name in test_dataloader_list:
        result_name = os.path.join(args.test_result,file_name+'.txt')
        if os.path.exists(result_name):
            print(result_name+'already exists!')
            continue
        print("Begin predict date",file_name)
        prediction, id = trainer.predict(test_dataloader)

        with open(result_name, "w", encoding="utf-8") as f:
            for index in range(len(prediction)):
                f.write("{}\t{}\n".format(id[index], prediction[index]))
        f.close()
    

    logging.info("Done!")

def main():
    predict(args)


if __name__ == '__main__':
    main()