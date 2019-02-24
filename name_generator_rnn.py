# -*- coding:utf-8 -*-
from __future__ import print_function, unicode_literals, division
from io import open
import glob
import os
import unicodedata
import string
import argparse

import torch
import torch.nn as nn
import random

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # plus EOS marker

category_line = {}
all_category = []
n_categories = 0
train_category = None

save_dir="data/save"

def parse():
    parser = argparse.ArgumentParser(description="rnn model for name generator")
    parser.add_argument('-it', '--iteration', type=int, default=100000, help="iterations of training")
    parser.add_argument('-p', '--print_every', type=int, default=5000, help="print the training result every iterations")
    parser.add_argument('-pl', '--plot_every', type=int, default=500, help="plotting the loss every iterations")
    parser.add_argument('-s', '--save_every', type=int, default=5000, help="save model params every iterations")
    parser.add_argument('-tr', '--train', action='store_true', help="Train the model with dataset")
    parser.add_argument('-te', '--test', action='store_true', help="test the saved model")
    parser.add_argument('-lm', '--load_model', help="load the saved model(e.g.model/name_generator_model_100000.tar)")
    parser.add_argument('-fn', '--filename', help="dataset file for training (e.g.data/names/*.txt)")
    parser.add_argument('-sl', '--single_letter', help="generate name with a letter, e.g. -sl A")
    parser.add_argument('-ml', '--multi_letters', help="generate names with letters, e.g. -ml ACD")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help="learning rate for training")
    parser.add_argument('-c', '--category', type=str, choices=['Arabic', 'Chinese', 'Czech', 'Dutch', 'English',
                                                               'French', 'German', 'Greek', 'Irish', 'Italian',
                                                               'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian',
                                                               'Scottish', 'Spanish', 'Vietnamese'],
                        help="language category to train or test")

    args = parser.parse_args()
    return args


# search specify file type
def findFiles(path):
    return glob.glob(path)

# turn unicode string to ascii plain, thanks to https://stackoverflow.com/a/518232/2809427
def Unicode2Ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if c in all_letters
        and "MN" != unicodedata.category(c))

# read line from file and split by '\n'
def readLines(filePath):
    lines = open(filePath, encoding="utf-8").read().strip().split('\n')
    return [Unicode2Ascii(line) for line in lines]

# create dataset from files
"""
args: filename with regular expression like data/names/*.txt
"""
def loadTrainingDataset(filenames):
    global category_line
    global all_category
    global n_categories
    for fileName in findFiles(filenames):
        category = os.path.splitext(os.path.basename(fileName))[0]
        all_category.append(category)
        lines = readLines(fileName)
        category_line[category] = lines

    n_categories = len(all_category)
    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

   # print(all_category)
    return category_line, all_category, n_categories

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, categories, input, hidden):
        in_combined = torch.cat((categories, input, hidden), dim=1)
        hidden = self.i2h(in_combined)
        output = self.i2o(in_combined)

        out_combined = torch.cat((output, hidden), dim=1)
        output = self.o2o(out_combined)

        output = self.softmax(self.dropout(output))
        return output, hidden

    def InitHidden(self):
        return torch.zeros(1, self.hidden_size)


# prepare data for training
# choose a item from list randomly
def randomChoice(l):
    return l[random.randint(0, len(l) -1)]

# choose training data pairs
def randomTrainingPairs(category=None):
    global train_category
    if category is None:
        category = randomChoice(all_category)
    train_category = category
    name = randomChoice(category_line[category])
    return category, name


# one-hot vector for category
def CategoryTensor(category):
    tensor = torch.zeros(1, n_categories)
    idx = all_category.index(category)
    tensor[0][idx] = 1
    return tensor

# one-hot matrix for input, ont include EOS
def InputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for idx in range(len(line)):
        letter = line[idx]
        tensor[idx][0][all_letters.find(letter)] = 1

    return tensor

# longTensor for second letter to EOS
def TargetTensor(line):
    letter_indexes = [all_letters.find(line[idx]) for idx in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # add index of EOS
    return torch.LongTensor(letter_indexes)

# make category, input and target tensors from random category, line pairs

def randomTrainingSample(category=None):
    category, line = randomTrainingPairs(category)
    category_tensor = CategoryTensor(category)
    input_line_tensor = InputTensor(line)
    target_line_tensor = TargetTensor(line)

    return category_tensor, input_line_tensor, target_line_tensor

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.InitHidden()
    rnn.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        per_loss = criterion(output, target_line_tensor[i])

        loss += per_loss

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-lr, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

def TimeCalulate(since):
    now = time.time()
    interval = now - since
    m = math.floor(interval/60)
    s = interval - 60 * m
    return "%dm %ds" %(m,s)

def runTrainingModel(n_iters=100000, print_every=5000, plot_every=500, save_every=5000, category=None, modelFile=None):
    all_losses = []
    total_loss = 0 # Reset every plot_every iters
    start = time.time()

    checkpoint = None
    start_iteration = 1
    if modelFile:
        checkpoint = torch.load(modelFile)
        rnn.load_state_dict(checkpoint["rnn"])
        start_iteration = checkpoint["iteration"]

    for iter in range(start_iteration, n_iters + 1):
        output, loss = train(*randomTrainingSample(category))
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (TimeCalulate(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append((total_loss / plot_every) if (iter - start_iteration >= plot_every) else loss)
            total_loss = 0

        if iter % save_every == 0:
            directory = os.path.join(save_dir, 'model')
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'rnn': rnn.state_dict(),
                'category': train_category,
                'loss': loss
            }, os.path.join(directory, '{}_{}.tar'.format('name_generator_model', iter)))

    return all_losses

# sample from a category and starting letter
def Sample(category, start_letter='A', modelFile=None, max_lenght = 20):
    if modelFile:
        checkpoint = torch.load(modelFile)
        rnn.load_state_dict(checkpoint["rnn"])
        if category is None:
            category = checkpoint["category"]

    hidden = rnn.InitHidden()
    category_tensor = CategoryTensor(category)
    input_tensor = InputTensor(start_letter)
    output_name = start_letter
    for i in range(max_lenght):
        output, hidden = rnn(category_tensor, input_tensor[0], hidden)
        topv, topi = output.topk(1)

        idx = topi[0][0]
        if idx == n_letters - 1: break
        else:
            letter = all_letters[idx]
            output_name += letter
            input_tensor = InputTensor(letter)
    return output_name

def Sampeles(category, start_letters="ABC", modelFile=None):
    names = []
    for letter in start_letters:
        names.append(Sample(category, letter, modelFile))
    return names

def run(args):
    modelFile = None
    if args.load_model:
        modelFile = args.load_model

    category = None
    if args.category:
        category = args.category
    if args.test:
        if modelFile is None:
            raise RuntimeError('Please choose a saved model to load')

        if args.single_letter:
            start_letter = args.single_letter
            print(Sample(category, start_letter, modelFile))
        elif args.multi_letters:
            print(Sampeles(category, args.multi_letters, modelFile))

        else:
            raise RuntimeError("please specify evaluate mode")

    elif args.train:
        runTrainingModel(category=category, modelFile=modelFile)

    else:
        raise RuntimeError("please specify running mode[test/train]")

if __name__=="__main__":

    args = parse()
    filename = "data/names/*.txt"
    if args.filename:
        filename = args.filename
    loadTrainingDataset(filename)

    criterion = nn.NLLLoss()
    lr = 0.0001
    if args.learning_rate:
        lr = args.learning_rate

    rnn = RNN(n_letters, 128, n_letters)
    run(args)