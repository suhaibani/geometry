from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from flair.embeddings import TransformerWordEmbeddings
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from flair.data import Sentence
from scipy.linalg import norm
import numpy as np
import argparse
import logging
import os
import pandas as pd
import torch.nn as nn
import torch
import json

def pretraining_params():
    config_class, model_class, tokenizer_class = {"bert": (BertConfig, BertModel, BertTokenizer)}
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config)

def dataset():
    # load dataset
    eval_data = load_and_cache_examples(args, tokenizer, labels, mode=args.file_name)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu access
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

def snli_embedding_fetch(file, pretrained_model, pooling, layer, v_norm, dest):
    def extract_sentences():
        sentence_pairs = []
        if file.endswith('.xlsx'):
            df = pd.read_excel(file)
            for i,j in zip(df['sentence1'][:100], df['sentence2'][:100]):
                sentence_pairs.append((i,j))
        elif file.endswith('.json'):
            t = open(file, 'r')
            for i in t.readlines():
                d = json.loads(i)
                sentence_pairs.append((d['sentence1'], d['sentence2']))
        return sentence_pairs

    def norm_search(vec):
        if v_norm.lower() == 'l1':
            return sum([np.abs(i) for i in vec])
        elif v_norm.lower() == 'l2':
            return np.sqrt(sum([np.abs(i)**2 for i in vec]))
        elif v_norm.lower() == 'linf':
            return np.max([np.abs(x) for x in vec])

    def pooling_sentence(sentence_tensors):
        vec = []
        for i in sentence_tensors:
            vect = i.embedding.cpu().detach().numpy()
            vec.append((norm(vect), vect))
        vec = sorted(vec, key=lambda x: x[0])
        if pooling.lower() == 'mean':
            #vec_mean = torch.zeros((len(vec), len(vec[0][1])))
            vec_len = len(vec[0][1])
            vec_mean = []
            for v in range(len(vec)):
                vec_mean.append(vec[v][1])
            vec_mean = torch.mean(torch.tensor(vec_mean), 0)
            return vec_mean.numpy()
        elif pooling.lower() == 'max':
            return vec[-1][1]
        elif pooling.lower() == 'min' :
            return vec[0][1]
        else:
            pass

    if pooling == 'default':
        e_S = TransformerDocumentEmbeddings(pretrained_model)
    elif pooling in ['max', 'min', 'mean']:
        e_S = TransformerWordEmbeddings(pretrained_model, layers=layer)

    l2_norm = []
    for pair in extract_sentences():
        sent_1 = Sentence(pair[0])
        sent_2 = Sentence(pair[1])
        e_S.embed(sent_1)
        e_S.embed(sent_2)
        if pooling == 'default':
            sent_1_e = sent_1.embedding.cpu().detach().numpy()
            sent_2_e = sent_2.embedding.cpu().detach().numpy()
        elif pooling in ['max', 'min', 'mean']:
            sent_1_e = pooling_sentence(sent_1)
            sent_2_e = pooling_sentence(sent_2)
        #print(sent_1_e.shape, type(sent_1_e))
        compare = norm_search(sent_1_e) > norm_search(sent_2_e)
        l2_norm.append((pair[0], pair[1], norm(sent_1_e), norm(sent_2_e), compare))

    l2_norm_frame = pd.DataFrame(l2_norm, columns=['sentence1', 'sentence2', 'sentence1_norm', 'sentence2norm', 'verdict'])
    print(l2_norm_frame.head(20))
    count = {}
    for i in l2_norm_frame['verdict']:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1
    with open(dest+'/performance.txt'.format(dest), 'a') as t_f:
        t_f.write('test_{}_{}_{}:-{}'.format(layer, pooling, v_norm, count))
        t_f.write('\n')
        t_f.close
    l2_norm_frame.to_csv(dest+'/test_{}_{}_{}.csv'.format(layer, pooling, v_norm))
    return l2_norm_frame

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    ## Required parameters
    p.add_argument("--data_dir", default=None, type=str, help="The input data dir. Should contain the training files.")
    p.add_argument("--file_name", default='test_entailment.xlsx', type=str, required=True, help="source file.")
    p.add_argument("--pretrained_model", default='bert-base-uncased')
    p.add_argument("--mode", default='train', type=str, help="feature dimension")
    p.add_argument("--pooling", default='mean', type=str, help="pooling operation")
    p.add_argument("--norm", default='l2', type=str, help="vector norm operation")
    p.add_argument("--layer", default='all', type=str, help="model layer")
    p.add_argument("--embedding_type", default=None, required=True, type=str, help='which embeddings to create')
    p.add_argument("--embedding_size", default=768, type=int, help='size of the embedding')
    p.add_argument("--dest_folder", default=None, required=True, help='where to store the embeddings created')

    args = p.parse_args()

    dest_folder= os.path.abspath(os.path.join(os.path.curdir, args.dest_folder))
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    snli_embedding_fetch(args.file_name,
                         args.pretrained_model,
                         args.pooling,
                         args.layer,
                         args.norm,
                         dest_folder)
