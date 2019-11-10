import sys
sys.path.append('SentimentClassifier')
from SentimentClassifier.core.w2vEmbReader import W2VEmbReader
from SentimentClassifier.core.reader import load_dataset
from SentimentClassifier.core.helper import pad_sequences
from collections import namedtuple
import torch
from torch import nn
from torch.utils.data import Dataset

def get_embeddings(vocab, folder='./SentimentClassifier', dim=100):
    emb_reader = W2VEmbReader(f'{folder}/word2vec/vectors/glove.6B.{dim}d.txt', emb_dim=dim)
    emb = nn.Embedding(len(vocab), dim)
    emb.weight = nn.Parameter(torch.tensor(emb_reader.get_emb_matrix_given_vocab(vocab, emb.weight.tolist())))
    return emb


class IMDbDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_data(folder='./SentimentClassifier', vocab_size=2000):
    acl_dir = f'{folder}/data/aclImdb'
    make_nt = lambda x: namedtuple('Args', x.keys())(**x)

    args = dict(train_path=acl_dir+'/train/',
                dev_path=acl_dir+'/valid/',
                test_path=acl_dir+'/test/',
                data_binary_path=None,
                vocab_size=vocab_size,
                out_dir_path=acl_dir+'/outta/',

                dropout_rate=0.5,
                cnn_layer=1,
                cnn_dim=250,
                cnn_window_size=3,

                seed=1337)

    ((train_x, train_y, train_filename_y),
    (dev_x, dev_y, dev_filename_y),
    (test_x, test_y, test_filename_y),
    vocab, vocab_size, overal_maxlen) = load_dataset(make_nt(args))

    train_dataset = IMDbDataSet(train_x, torch.FloatTensor(train_y))
    dev_dataset = IMDbDataSet(dev_x, torch.FloatTensor(dev_y))
    test_dataset = IMDbDataSet(test_x, torch.FloatTensor(test_y))

    return train_dataset, dev_dataset, test_dataset, vocab
