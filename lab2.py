# %%
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from jiwer import cer,wer
import random
import re
import pickle
from torch.nn.utils.rnn import pad_sequence

from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

# %%
data_path = 'data/combined.xml'
tree = ET.parse(data_path)
root = tree.getroot()

# %%
phoneme_list = ["<sos>", "<eos>"]
allophone_list = ["<sos>", "<eos>"]
g2p = []

# %%
for sent_idx in tqdm(range(0, len(root))):
    for tag_idx in range(len(root[sent_idx])):
        tag = root[sent_idx][tag_idx].tag
        attrib = root[sent_idx][tag_idx].attrib
        if tag == "word":
            if "original" in attrib and attrib["original"] != '':
                phoneme = ''
                grapheme = ''
                allophone = ''
                for child2 in root[sent_idx][tag_idx]:
                    if child2.tag == "phoneme":
                        if len(phoneme) == 0:
                            phoneme += child2.attrib['ph']
                        else:
                            phoneme += ' ' + child2.attrib['ph']
                        if child2.attrib['ph'] not in phoneme_list:
                            phoneme_list.append(child2.attrib['ph'])
                    if child2.tag == "letter":
                        try:
                            grapheme += child2.attrib['char']
                        except:
                            grapheme += ''
                    if child2.tag == "allophone":
#                         alph = re.sub(r'\d+', r'', child2.attrib['ph']).strip().lower()
                        alph = child2.attrib['ph']
                        if alph not in allophone_list:
                            allophone_list.append(alph)
                        allophone += alph
                if [grapheme, allophone] not in g2p:
                    g2p.append([grapheme, allophone])
    
    # with open('alophones3.json', 'w') as f:
    #     json.dump(allophone_list, f)

    # with open('words4.json', 'w', encoding='UTF-8') as f:
    #     json.dump(g2p, f, indent=4)

# %%
# def prepare_data(org_data):
# #     f = open(filename)
# #     data = [x.strip('\n').split('\t') for x in f.readlines()]
# #     data = [(list(x[0]), x[1].split(' ')) for x in org_data]
#     data = [(x[0], x[1].replace(' ', '')) for x in org_data]
#     return data

# # with open('../input/laba2-g2p/dict_g2p.json', 'r') as f:
# #     data = json.load(f)
# # g2p = prepare_data(data)

# %%
data = pd.DataFrame(g2p, columns=['grapheme', 'phoneme'])
print(len(data))
data.head()

# %%
graphemes=list(data["grapheme"])
phonemes=list(data["phoneme"])
len(graphemes),len(phonemes)

# %%
all_data=[]
for i in range(len(graphemes)):
    all_data.append((graphemes[i],phonemes[i]))
all_data[:10]

# %%
train_iter,test_iter = train_test_split(all_data,test_size=0.1,random_state=42,shuffle=True)
len(train_iter),len(test_iter)

# %%
def my_tokenizer(word : str):
    return list(word)

# %%
token_transform = {}
vocab_transform = {}

SRC_LANGUAGE = 'grapheme'
TGT_LANGUAGE = 'phoneme'

token_transform[SRC_LANGUAGE] = my_tokenizer
token_transform[TGT_LANGUAGE] = my_tokenizer

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    all_iter=all_data
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(all_data, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# %%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE

# %%
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# %%
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# %%
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

# %%
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# %%
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], 
                                               vocab_transform[ln], 
                                               tensor_transform) 


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# %%
def train_function(model, optimizer):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in tqdm(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)

# %%
def evaluate_function(model):
    model.eval()
    losses = 0

    val_iter=test_iter
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

# # %%
# torch.manual_seed(42)

# SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
# TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
# EMB_SIZE = 512
# NHEAD = 8
# FFN_HID_DIM = 512
# BATCH_SIZE = 128
# NUM_ENCODER_LAYERS = 5
# NUM_DECODER_LAYERS = 5

# g2p_model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
#                                  NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

# for p in g2p_model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

# g2p_model = g2p_model.to(DEVICE)

# loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
# optimizer = torch.optim.Adam(g2p_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# # %%
# from timeit import default_timer as timer
# NUM_EPOCHS = 25

# stats_for_plotting={"train_losses":[],"test_losses":[]}
# for epoch in range(1, NUM_EPOCHS+1):
#     start_time = timer()
#     train_loss = train_function(g2p_model, optimizer)
#     end_time = timer()
#     val_loss = evaluate_function(g2p_model)
#     print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
#     stats_for_plotting["train_losses"].append(train_loss)
#     stats_for_plotting["test_losses"].append(val_loss)
#     torch.save(g2p_model.state_dict(),f"phonetics/best_model.pth")

# # %%
# plt.plot(stats_for_plotting["train_losses"],label="train")
# plt.plot(stats_for_plotting["test_losses"],label="validation")
# plt.legend(loc="upper right")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

# %%
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# %%
def inference(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

# # %%
# from tqdm import tqdm
# all_preds=[]
# for gs,ps in tqdm(test_iter):
#     ground_truth=ps
#     myoutput=inference(g2p_model,gs)
#     all_preds.append(myoutput.replace(" ",""))

# # %%
# trues=0
# all_num=0
# average_cer=0
# average_wer=0
# for i in range(len(all_preds)):
#     if all_preds[i]==test_iter[i][1]:
#         trues+=1
#     all_num+=1
#     average_cer+=cer(test_iter[i][1],all_preds[i])
#     average_wer+=wer(test_iter[i][1],all_preds[i])
# average_cer/=len(all_preds)
# average_wer/=len(all_preds)
# exact_accuracy=trues/all_num
# print(f"Exact Accuracy : {exact_accuracy}")
# print(f"PER : {average_cer}")
# print(f"WER : {average_wer}")

# # %%
# some_example_df = pd.DataFrame()
# some_examples=random.sample(test_iter, 20)
# example_gs=[]
# example_gt_phonemes=[]
# example_pred_phonemes=[]
# for example in some_examples:
#     example_gs.append(example[0])
#     example_gt_phonemes.append(example[1])
#     example_pred_phonemes.append(inference(g2p_model,example[0]).replace(" ",""))
# some_example_df["Grapheme"]=example_gs
# some_example_df["Predicted Phoneme"]=example_pred_phonemes
# some_example_df["Correct Phoneme"]=example_gt_phonemes
# some_example_df

# # %% [markdown]
# # # Построение классификатора для глассных

# # %% [markdown]
# # Загрузим модель

# # %%
# torch.manual_seed(42)

# SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
# TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
# EMB_SIZE = 512
# NHEAD = 8
# FFN_HID_DIM = 512
# BATCH_SIZE = 128
# NUM_ENCODER_LAYERS = 5
# NUM_DECODER_LAYERS = 5

# g2p_model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
#                                  NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

# for p in g2p_model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

# g2p_model.load_state_dict(torch.load('phonetics/best_model.pth', map_location=torch.device('cpu')))
# g2p_model = g2p_model.to(DEVICE)

# # %% [markdown]
# # Загрузим и обработаем исходнные данные. Создадим df с признаками, целевой меткой будет являться ударение гласной и сама аллофона  

# # %%
# phoneme_vowel = ['y', 'i', 'o', 'a', 'u', 'e']
# phoneme_consonant = ['sh', 's', 't', 'l', 'k', 'r', 'n', 'j', 'b', 'z','v','g','h','d','ch','f','m','zh','c','p','sc']
# grapheme_vowel = ["а", "е", "ё", "и", "о", "у", "ы", "э", "ю", "я"]
# grapheme_consonant = ["б", "в", "г", "д", "ж", "з", "й", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ь"]

# data_path = 'data/combined.xml'
# tree = ET.parse(data_path)
# root = tree.getroot()

# # %%
# dictitem = ['subpart_of_speech']
# content = ['PunktEnd', 'PunktBeg', 'EmphEnd', 'EmphBeg']
# phoneme_vowel = ['y', 'i', 'o', 'a', 'u', 'e']
# phoneme_consonant = ['sh', 's', 't', 'l', 'k', 'r', 'n', 'j', 'b', 'z','v','g','h','d','ch','f','m','zh','c','p','sc']
# grapheme_vowel = ["а", "е", "ё", "и", "о", "у", "ы", "э", "ю", "я"]
# grapheme_consonant = ["б", "в", "г", "д", "ж", "з", "й", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ь"]
# doubling = ['sh', 'sc', 'zh', 'ch']

# dct_feature = {
#                 'phoneme_previous': [],# предыдущая фонема +
#                 'phoneme_current': [], # текущая фонема +
#                 'phoneme_next': [], # следующая фонема +
#                 'pos_phoneme_start': [], # позиция текущей фонемы от начала слова + 
#                 'pos_phoneme_end': [], # позиция текущей фонемы от конца слова +
#                 'flag_letter_previous': [], # ударность предыдущей фонемы +
#                 'flag_letter_current': [], # ударность текущей фонемы +
#                 'flag_letter_next': [], # ударность последующей фонемы +
#                 'subpart_of_speech_current': [], # часть речи текущего слова +
#                 'pos_word_start_sent': [], # позиция текущего слова в предложении от начала +
#                 'pos_word_end_sent': [], # позиция текущего слова в предложении от конца +
#                 'nucleus': [], # ударность слова +
#                 'pause': [],
#                 'pause_type': [],
#                 'letter_previous': [],
#                 'letter_current': [],
#                 'letter_next': [],
#                 'word': [],
#                 'phoneme_word': [],
#                 'num_word_before': [],
#                 'sentence_num': [],
#                 'phonem_place': [],
#                 'allophone_word': [],
#                 'allophone': [],
#                 }

    
# for sent_idx in tqdm(range(0, len(root))):
#     num_word_before = 0
#     for tag_idx in range(len(root[sent_idx])):
#         tag = root[sent_idx][tag_idx].tag
#         attrib = root[sent_idx][tag_idx].attrib
#         if tag == "word":
#             if "original" in attrib and attrib["original"] != '':
#                 first_dictitem = True
#                 letter_list = []
#                 flag_letter_list = []
#                 allophone_list = []
#                 allo_cnt = 0
#                 flag_allo_cnt = True
#                 for child2 in root[sent_idx][tag_idx]:
#                     if child2.tag == "dictitem":
#                         if first_dictitem:                         
#                             for feature_dictitem in dictitem:
#                                 if feature_dictitem in child2.keys():
#                                     subpart_of_speech_current = int(child2.attrib[feature_dictitem])
#                                 else:
#                                     subpart_of_speech_current = np.nan
#                         first_dictitem = False
#                     if child2.tag == "letter":
#                         try:
#                             letter = child2.attrib['char']
#                         except:
#                             letter = ''
#                         try:
#                             flag_letter = child2.attrib['flag']
#                         except:
#                             flag_letter = "0"
#                         letter_list.append(letter)
#                         flag_letter_list.append(flag_letter)
#                     if child2.tag == "allophone":
#                         allophone = child2.attrib['ph']
#                         if flag_allo_cnt and re.sub(r'\d+', r'', allophone) in phoneme_vowel:
#                             idx_g_prev = allo_cnt
#                             flag_allo_cnt = False
#                         allophone_list.append(allophone)
#                         allo_cnt += 1
#                 word_original = attrib["original"]
#                 if flag_allo_cnt:
#                     idx_g_prev = 0
#                 phoneme_predict = inference(g2p_model, ''.join(letter_list)).replace(' \'', '\'').replace('s h', 'sh').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')[1:-1]
#                 pos_phoneme_start = 0
#                 for idx_p, p in enumerate(phoneme_predict):
#                     if p in phoneme_vowel:
#                         dct_feature['word'].append(attrib['original'])
#                         dct_feature['phoneme_word'].append("".join(phoneme_predict))
#                         dct_feature['pos_word_start_sent'].append(tag_idx)
#                         dct_feature['pos_word_end_sent'].append(len(root[sent_idx])-1-tag_idx)
#                         dct_feature['subpart_of_speech_current'].append(subpart_of_speech_current)
#                         if "nucleus" in attrib:
#                             if attrib["nucleus"] == "2":
#                                 dct_feature['nucleus'].append(1)
#                             else:
#                                 dct_feature['nucleus'].append(0)
#                         else:
#                             dct_feature['nucleus'].append(0)
#                         if idx_p == 0:
#                             dct_feature['phoneme_previous'].append('<sos>')
#                         else:                           
                            
#                             dct_feature['phoneme_previous'].append(text_transform[TGT_LANGUAGE](phoneme_predict[idx_p-1])[1].item())
#                         if idx_p == len(phoneme_predict)-1:
#                             dct_feature['phoneme_next'].append('<eos>')
#                         else:
#                             dct_feature['phoneme_next'].append(text_transform[TGT_LANGUAGE](phoneme_predict[idx_p+1])[1].item())
#                         dct_feature['phoneme_current'].append(text_transform[TGT_LANGUAGE](phoneme_predict[idx_p])[1].item())
#                         dct_feature['pos_phoneme_start'].append(pos_phoneme_start)
#                         pos_phoneme_end = len(phoneme_predict) - 1 - idx_p
#                         dct_feature['pos_phoneme_end'].append(pos_phoneme_end)     
#                         try:
#                             allp = re.sub(r'\d+', r'', allophone_list[idx_p]).strip()
#                         except:
#                             allp = np.nan
# #                         if (idx_p <= len(letter_list)-1 and (len(letter_list) == len(phoneme_predict) or letter_list[idx_p] in grapheme_vowel)) or allp in phoneme_vowel:
#                         if allp in phoneme_vowel or allp==p:
#                             idx_g = idx_p
#                         else:
# #                             idx_g = idx_p
# #                             grapheme_predict = ''
# #                             while grapheme_predict not in grapheme_vowel and idx_g >= 0:
# #                                 idx_g -= 1
# #                                 if idx_g <= len(letter_list) - 1:
# #                                     grapheme_predict = letter_list[idx_g]    
#                             idx_g = idx_p
#                             grapheme_predict = ''
#                             while grapheme_predict not in phoneme_vowel and idx_g >= 0:
#                                 idx_g -= 1
#                                 if idx_g <= 0:
#                                     break
#                                 if idx_g <= len(allophone_list) - 1:
#                                     grapheme_predict = re.sub(r'\d+', r'', allophone_list[idx_g]).strip()

# #                         if ''.join(letter_list)=='бесчестными':
# #                             print(letter_list)
# #                             print(allophone_list)
# #                             print(phoneme_predict)
# #                             print('idx_p', idx_p)
# #                             print('idx_g', idx_g)
# #                             print('p', p)
# #                             print(allophone_list[idx_g])
# #                             print('___________________')
                        
#                         idx_letter = idx_g if 'j' not in allophone_list[:idx_g] else idx_g-(allophone_list[:idx_g]).count('j')
#                         for db in doubling:
#                             if db in allophone_list[:idx_g]:
#                                 idx_letter = idx_letter - (allophone_list[:idx_g]).count(db)
                            
                        
#                         if idx_g == 0:
#                             dct_feature['flag_letter_previous'].append("-1")
#                             dct_feature['letter_previous'].append('<sos>')
#                         else:
#                             dct_feature['flag_letter_previous'].append(flag_letter_list[idx_letter-1])
#                             dct_feature['letter_previous'].append(text_transform[SRC_LANGUAGE](letter_list[idx_letter-1])[1].item())
# #                         print(idx_g, len(letter_list)-1)
#                         if idx_g >= len(letter_list)-1:
#                             dct_feature['flag_letter_next'].append("-1")
#                             dct_feature['letter_next'].append('<eos>')
#                         else:
#                             dct_feature['flag_letter_next'].append(flag_letter_list[idx_letter+1])
#                             dct_feature['letter_next'].append(text_transform[SRC_LANGUAGE](letter_list[idx_letter+1])[1].item())
#                         dct_feature['flag_letter_current'].append(flag_letter_list[idx_letter])
#                         dct_feature['letter_current'].append(text_transform[SRC_LANGUAGE](letter_list[idx_letter])[1].item())
#                         dct_feature['allophone_word'].append(''.join(allophone_list))
#                         try:
#                             dct_feature['allophone'].append(allophone_list[idx_g])
#                         except:
#                             try:
#                                 dct_feature['allophone'].append(allophone_list[idx_g_prev])
#                             except:
#                                 dct_feature['allophone'].append(np.nan)
#                         if root[sent_idx][tag_idx+2].tag == "pause":
#                             dct_feature['pause'].append(1)
#                             dct_feature['pause_type'].append(root[sent_idx][tag_idx+2].attrib["type"])
#                         else:
#                             dct_feature['pause_type'].append('no')
#                             dct_feature['pause'].append(0)
#                         dct_feature['sentence_num'].append(sent_idx)
#                         dct_feature['num_word_before'].append(num_word_before)
#                         dct_feature['phonem_place'].append(idx_p)
#                         idx_g_prev = idx_g
#                     pos_phoneme_start += 1
                    
#                 num_word_before += 1
                
# with open('phonetics/df_classifier_data_all.pickle', 'wb') as f:
#     pickle.dump(dct_feature, f, protocol=pickle.HIGHEST_PROTOCOL)

# # %%
# with open('phonetics/df_classifier_data_all.pickle', 'rb') as f:
#     data = pickle.load(f)

# # %%
# df = pd.DataFrame(data)
# df.drop(['sentence_num', 'num_word_before', 'pos_phoneme_start'], inplace=True, axis=1)
# df.head()

# # %%
# df.loc[df[df['phoneme_previous']=='<sos>'].index, 'phoneme_previous'] = vocab_transform[TGT_LANGUAGE]['<bos>']
# df.loc[df[df['phoneme_previous']=='<eos>'].index, 'phoneme_previous'] = vocab_transform[TGT_LANGUAGE]['<eos>']
# df.loc[df[df['phoneme_next']=='<sos>'].index, 'phoneme_next'] = vocab_transform[TGT_LANGUAGE]['<bos>']
# df.loc[df[df['phoneme_next']=='<eos>'].index, 'phoneme_next'] = vocab_transform[TGT_LANGUAGE]['<eos>']

# df.loc[df[df['letter_previous']=='<sos>'].index, 'letter_previous'] = vocab_transform[TGT_LANGUAGE]['<bos>']
# df.loc[df[df['letter_previous']=='<eos>'].index, 'letter_previous'] = vocab_transform[TGT_LANGUAGE]['<eos>']
# df.loc[df[df['letter_next']=='<sos>'].index, 'letter_next'] = vocab_transform[TGT_LANGUAGE]['<bos>']
# df.loc[df[df['letter_next']=='<eos>'].index, 'letter_next'] = vocab_transform[TGT_LANGUAGE]['<eos>']

# # %%
# df.drop('allophone_word', inplace=True, axis=1)

# # %%
# df['allophone'].value_counts()

# # %%
# df = df.loc[(df['allophone'] != 's')&(df['allophone'] != 'f')]

# # %%
# vowel_allophone, nuclear = [], []
# for x in tqdm(df.allophone):
#     vowel_allophone.append(list(x)[0])
#     nuclear.append(list(x)[1])
# df.drop('allophone', inplace=True, axis=1)

# # %%
# df.columns

# # %%
# df.head()

# # %%
# df.reset_index(drop=True, inplace=True)

# # %%
# df = df.astype({'letter_current': int})

# # %% [markdown]
# # ### Построение классификатора гласных букв

# # %%
# # from sklearn import preprocessing
# # le = preprocessing.LabelEncoder()
# # labels_vowel = le.fit_transform(vowel_allophone)

# # %%
# y = np.array([vocab_transform[TGT_LANGUAGE][x] for x in vowel_allophone])

# # %%
# from collections import Counter
# Counter(y)

# %%
def catboost_GridSearchCV(X, y, params, cat_features, n_splits=2):
    ps = {'f1':0,
          'param': []
    }
    
    predict=None
    
    for prms in tqdm(list(ParameterGrid(params)), ascii=True, desc='Params Tuning:'):
                          
        f1 = cross_val(X, y, prms, cat_features, n_splits=n_splits)

        if f1>ps['f1']:
            ps['f1'] = f1
            ps['param'] = prms
    print('F1: '+str(ps['f1']))
    print('Params: '+str(ps['param']))
    
    return ps['param']

# %%
def cross_val(X, y, param, cat_features, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    f1_list = []
    predict = None
    
    for tr_ind, val_ind in skf.split(X, y):
#         print(tr_ind)
        X_train = X.loc[tr_ind]
        y_train = y[tr_ind]
        
        X_valid = X.loc[val_ind]
        y_valid = y[val_ind]
        
        clf = CatBoostClassifier(iterations=500,
                                loss_function = param['loss_function'],
                                depth=param['depth'],
                                l2_leaf_reg = param['l2_leaf_reg'],
                                eval_metric = 'TotalF1',
                                leaf_estimation_iterations = 10,
                                use_best_model=True,
                                logging_level='Silent',
#                                 class_weights =dct_weights,
                                task_type="GPU",
                                devices='0',
                                learning_rate=1
        )
        
        clf.fit(X_train, 
                y_train,
                cat_features=cat_features,
                eval_set=(X_valid, y_valid)
        )
        
        y_pred = clf.predict(X_valid)
        f1 = f1_score(y_valid, y_pred, average='weighted')
        f1_list.append(f1)
    return sum(f1_list)/n_splits

# # %%
# # cat_features=['pause_type', 'word', 'phoneme_word']
# cat_features=['pause_type', 'word', 'phoneme_word', 'phoneme_previous', 'phoneme_current', 'phoneme_next', 'flag_letter_previous',
#              'flag_letter_current', 'subpart_of_speech_current', 'nucleus', 'pause', 'letter_previous', 'letter_current', 'letter_next']

# # %%
# params = {'depth':[2, 3, 4],
#           'loss_function': ['MultiClassOneVsAll', 'MultiClass'],
#           'l2_leaf_reg':np.logspace(-20, -19, 3)
# }

# param = catboost_GridSearchCV(df, y, params, cat_features)
# print(param)

# # %%
# X_train, X_test, y_train, y_test = train_test_split(df,
#                                                     y, 
#                                                     shuffle=True,
#                                                     random_state=42,
#                                                     train_size=0.9,
#                                                     stratify=y
#                                                    )

# # %%
# from catboost import CatBoostClassifier
# from sklearn.model_selection import train_test_split

# clf = CatBoostClassifier(iterations=3500,
#                         loss_function = param['loss_function'],
#                         depth=param['depth'],
#                         l2_leaf_reg = param['l2_leaf_reg'],
#                         eval_metric = 'TotalF1',
#                         leaf_estimation_iterations = 10,
#                         use_best_model=True,
#                         task_type="GPU",
# #                         class_weights =(1, 2.89),
#                         devices='0',
#                         metric_period=250
#                         )

# # %%
# clf.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True, cat_features=cat_features)

# # %%
# clf.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True, cat_features=cat_features)

# # %%
# y_pred = clf.predict(X_test)

# # %%
# target_names = [vocab_transform[TGT_LANGUAGE].lookup_tokens([x])[0] for x in np.unique(y)]

# # %%
# print(classification_report(y_test, y_pred, target_names=target_names))

# # %%
# cm = confusion_matrix(y_test, y_pred)
# df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
# # show_confusion_matrix(df_cm)
# plt.figure(figsize=(10,8))
# hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize":18})
# hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
# hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
# plt.ylabel('True sentiment', fontsize=18)
# plt.xlabel('Predicted sentiment', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=18)

# # %%
# with open('phonetics/clf_vowel_allophones_2.pkl', 'wb') as f:
#     pickle.dump(clf, f)

# # %% [markdown]
# # ### Классификация ударений гласных

# # %%
# y = np.array([int(x) for x in nuclear])

# # %%
# dct_weights = {}
# cnt = Counter(y)
# max_support = max(cnt.values())
# for k, v in cnt.items():
#     dct_weights[k] = max_support / v
# dct_weights

# # %%
# params = {'depth':[2, 3, 4],
#           'loss_function': ['MultiClassOneVsAll', 'MultiClass'],
#           'l2_leaf_reg':np.logspace(-20, -19, 3)
# }

# param = catboost_GridSearchCV(df, y, params, cat_features)
# print(param)

# # %%
# params = {'depth':[2, 3, 4],
#           'loss_function': ['MultiClassOneVsAll', 'MultiClass'],
#           'l2_leaf_reg':np.logspace(-20, -19, 3)
# }

# param = catboost_GridSearchCV(df, y, params, cat_features)
# print(param)

# # %%
# X_train, X_test, y_train, y_test = train_test_split(df,
#                                                     y, 
#                                                     shuffle=True,
#                                                     random_state=42,
#                                                     train_size=0.9,
#                                                     stratify=y
#                                                    )

# # %%
# clf_nuclear = CatBoostClassifier(iterations=3500,
#                         loss_function = param['loss_function'],
#                         depth=param['depth'],
#                         l2_leaf_reg = param['l2_leaf_reg'],
#                         eval_metric = 'TotalF1',
#                         leaf_estimation_iterations = 10,
#                         use_best_model=True,
#                         task_type="GPU",
#                         class_weights = dct_weights,
#                         devices='0',
#                         metric_period=250
#                         )

# # %%
# clf_nuclear.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True, cat_features=cat_features)

# # %%
# y_pred = clf_nuclear.predict(X_test)

# # %%
# print(classification_report(y_test, y_pred))

# # %%
# cm = confusion_matrix(y_test, y_pred)
# df_cm = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))
# # show_confusion_matrix(df_cm)
# plt.figure(figsize=(10,8))
# hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize":18})
# hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
# hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
# plt.ylabel('True sentiment', fontsize=18)
# plt.xlabel('Predicted sentiment', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=18)

# # %%
# with open('phonetics/clf_vowel_allophones_nuclear_2.pkl', 'wb') as f:
#     pickle.dump(clf_nuclear, f)

# # %% [markdown]
# # ## Построение классификатора для конечных аллофонов "H", "CH", "C", "SC"

# # %%
# with open('phonetics/df_classifier_data_all.pickle', 'rb') as f:
#     data = pickle.load(f)
# data_path = 'data/combined.xml'
# tree = ET.parse(data_path)
# root = tree.getroot()

# # %%
# df = pd.DataFrame(data)

# # %%
# end_phoneme_upper = ["H", "CH", "C", "SC", "h", 'ch', 'c', 'sc']

# # %%
# idx_ph_list = []
# for idx, phoneme in tqdm(df['allophone_word'].items(), total=len(df)):
#     end_ph = ' '.join(list(phoneme)).replace('s h', 'sh').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')[-1]
#     if end_ph in end_phoneme_upper:
#         idx_ph_list.append(idx)

# # %%
# dictitem = ['subpart_of_speech']
# content = ['PunktEnd', 'PunktBeg', 'EmphEnd', 'EmphBeg']
# phoneme_vowel = ['y', 'i', 'o', 'a', 'u', 'e']
# phoneme_consonant = ['sh', 's', 't', 'l', 'k', 'r', 'n', 'j', 'b', 'z','v','g','h','d','ch','f','m','zh','c','p','sc']
# grapheme_vowel = ["а", "е", "ё", "и", "о", "у", "ы", "э", "ю", "я"]
# grapheme_consonant = ["б", "в", "г", "д", "ж", "з", "й", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ь"]
# doubling = ['sh', 'sc', 'zh', 'ch']

# dct_feature = {
#                 'phoneme_previous': [],# предыдущая фонема +
#                 'phoneme_current': [], # текущая фонема +
#                 'phoneme_next': [], # следующая фонема +
# #                 'pos_phoneme_start': [], # позиция текущей фонемы от начала слова + 
# #                 'pos_phoneme_end': [], # позиция текущей фонемы от конца слова +
#                 'flag_letter_previous': [], # ударность предыдущей фонемы +
#                 'flag_letter_current': [], # ударность текущей фонемы +
#                 'flag_letter_next': [], # ударность последующей фонемы +
#                 'subpart_of_speech_current': [], # часть речи текущего слова +
#                 'pos_word_start_sent': [], # позиция текущего слова в предложении от начала +
#                 'pos_word_end_sent': [], # позиция текущего слова в предложении от конца +
#                 'nucleus': [], # ударность слова +
#                 'pause': [],
#                 'pause_type': [],
#                 'letter_previous': [],
#                 'letter_current': [],
#                 'letter_next': [],
#                 'word': [],
#                 'phoneme_word': [],
#                 'num_word_before': [],
#                 'sentence_num': [],
#                 'phonem_place': [],
#                 'allophone_word': [],
#                 'allophone': [],
#                 }

    
# for sent_idx in tqdm(np.unique(df.loc[idx_ph_list].sentence_num)):
#     num_word_before = 0
#     for tag_idx in range(len(root[sent_idx])):
#         tag = root[sent_idx][tag_idx].tag
#         attrib = root[sent_idx][tag_idx].attrib
#         if tag == "word":
#             if "original" in attrib and attrib["original"] != '':
#                 first_dictitem = True
#                 letter_list = []
#                 flag_letter_list = []
#                 allophone_list = []
#                 allo_cnt = 0
#                 flag_allo_cnt = True
#                 for child2 in root[sent_idx][tag_idx]:
#                     if child2.tag == "dictitem":
#                         if first_dictitem:                         
#                             for feature_dictitem in dictitem:
#                                 if feature_dictitem in child2.keys():
#                                     subpart_of_speech_current = int(child2.attrib[feature_dictitem])
#                                 else:
#                                     subpart_of_speech_current = np.nan
#                         first_dictitem = False
#                     if child2.tag == "letter":
#                         letter = child2.attrib['char']
#                         try:
#                             flag_letter = child2.attrib['flag']
#                         except:
#                             flag_letter = "0"
#                         letter_list.append(letter)
#                         flag_letter_list.append(flag_letter)
#                     if child2.tag == "allophone":
#                         allophone = child2.attrib['ph']
#                         if flag_allo_cnt and re.sub(r'\d+', r'', allophone) in phoneme_vowel:
#                             idx_g_prev = allo_cnt
#                             flag_allo_cnt = False
#                         allophone_list.append(allophone)
#                         allo_cnt += 1
#                 word_original = attrib["original"]
#                 if flag_allo_cnt:
#                     idx_g_prev = 0
                
#                 if len(df[(df.sentence_num==sent_idx)&(df.word==word_original)])==0:
#                     break
                
#                 phoneme_predict = ' '.join(list(df[(df.sentence_num==sent_idx)&(df.word==word_original)].phoneme_word)[0]).replace(' \'', '\'').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')
            
#                 try:
#                     idx_next_word = list(df[(df.sentence_num==sent_idx)&(df.word==word_original)].index)[-1]+1
#                     phoneme_next = ' '.join(list(df.loc[idx_next_word].phoneme_word)[0]).replace('s h', 'sh').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')
#                     letter_next = list(df.loc[idx_next_word].word)[0]
#                 except:
#                     phoneme_predict = np.nan
#                     letter_next = np.nan

#                 if phoneme_predict[-1] in end_phoneme_upper:     
#                     dct_feature['word'].append(attrib['original'])
#                     dct_feature['phoneme_word'].append("".join(phoneme_predict))
#                     dct_feature['pos_word_start_sent'].append(tag_idx)
#                     dct_feature['pos_word_end_sent'].append(len(root[sent_idx])-1-tag_idx)
#                     dct_feature['subpart_of_speech_current'].append(subpart_of_speech_current)
#                     if "nucleus" in attrib:
#                         if attrib["nucleus"] == "2":
#                             dct_feature['nucleus'].append(1)
#                         else:
#                             dct_feature['nucleus'].append(0)
#                     else:
#                         dct_feature['nucleus'].append(0)
                    
                         
#                     dct_feature['phoneme_previous'].append(text_transform[TGT_LANGUAGE](phoneme_predict[-2])[1].item())

       
#                     dct_feature['phoneme_next'].append(text_transform[TGT_LANGUAGE](phoneme_next)[1].item())
        
        
#                     dct_feature['phoneme_current'].append(text_transform[TGT_LANGUAGE](phoneme_predict[-1])[1].item())
            
#                     pos_phoneme_end = len(phoneme_predict)
    
#                     dct_feature['flag_letter_previous'].append(flag_letter_list[-2])
#                     dct_feature['letter_previous'].append(text_transform[SRC_LANGUAGE](letter_list[-2])[1].item())
#                     dct_feature['letter_next'].append(letter_next)
                                                      
#                     dct_feature['flag_letter_current'].append(flag_letter_list[-1])
#                     dct_feature['letter_current'].append(text_transform[SRC_LANGUAGE](letter_list[-1])[1].item())
#                     dct_feature['allophone_word'].append(''.join(allophone_list))
    
#                     dct_feature['allophone'].append(allophone_list[-1])

#                     if root[sent_idx][tag_idx+2].tag == "pause":
#                         dct_feature['pause'].append(1)
#                         dct_feature['pause_type'].append(root[sent_idx][tag_idx+2].attrib["type"])
#                     else:
#                         dct_feature['pause_type'].append('no')
#                         dct_feature['pause'].append(0)
#                     dct_feature['sentence_num'].append(sent_idx)
#                     dct_feature['num_word_before'].append(num_word_before)
#                     dct_feature['phonem_place'].append(pos_phoneme_end)
                    
#             num_word_before += 1
                
# # with open('df_classifier_data_all.pickle', 'wb') as f:
# #     pickle.dump(dct_feature, f, protocol=pickle.HIGHEST_PROTOCOL)


# # %%
# with open('phonetics/df_classifier_data_all.pickle', 'rb') as f:
#     data = pickle.load(f)

# # %%
# df = pd.DataFrame(data)
# df['allophone'].value_counts()
# df = df.astype({'letter_current': int})

# # %%
# y = df['allophone']

# # %%
# # from sklearn import preprocessing
# # le = preprocessing.LabelEncoder()
# # y = le.fit_transform(df['allophone'])

# # %%
# dct_weights = {}
# cnt = Counter(y)
# max_support = max(cnt.values())
# for k, v in cnt.items():
#     dct_weights[k] = max_support / v
# dct_weights

# # %%
# df.columns

# # %%
# df.drop(['sentence_num', 'allophone_word', 'allophone'], axis=1, inplace=True)

# # %%
# cat_features=['pause_type', 'word', 'phoneme_word', 'phoneme_previous', 'phoneme_current', 'phoneme_next', 'flag_letter_previous',
#              'flag_letter_current', 'subpart_of_speech_current', 'nucleus', 'pause', 'letter_previous', 'letter_current', 'letter_next']

# # %%
# params = {'depth':[2, 3, 4],
#           'loss_function': ['MultiClassOneVsAll', 'MultiClass'],
#           'l2_leaf_reg':np.logspace(-20, -19, 3)
# }

# param = catboost_GridSearchCV(df, y, params, cat_features)
# print(param)


# X_train, X_test, y_train, y_test = train_test_split(df,
#                                                     y, 
#                                                     shuffle=True,
#                                                     random_state=42,
#                                                     train_size=0.9,
#                                                     stratify=y
#                                                    )

# # %%
# clf_ends_upper_allophone = CatBoostClassifier(iterations=3500,
#                         loss_function = param['loss_function'],
#                         depth=param['depth'],
#                         l2_leaf_reg = param['l2_leaf_reg'],
#                         eval_metric = 'TotalF1',
#                         leaf_estimation_iterations = 10,
#                         use_best_model=True,
#                         task_type="GPU",
#                         class_weights = dct_weights,
#                         devices='0',
#                         metric_period=250
#                         )

# # %%
# clf_ends_upper_allophone.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True, cat_features=cat_features)

# # %%
# y_pred = clf_ends_upper_allophone.predict(X_test)

# # %%
# print(classification_report(y_test, y_pred))

# # %%
# Counter(y)

# # %%
# cm = confusion_matrix(y_test, y_pred)
# df_cm = pd.DataFrame(cm)
# # show_confusion_matrix(df_cm)
# plt.figure(figsize=(10,8))
# hmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize":18})
# hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
# hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
# plt.ylabel('True sentiment', fontsize=18)
# plt.xlabel('Predicted sentiment', fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=18)

# # %%
# with open('phonetics/clf_ends_upper_allophone.pkl', 'wb') as f:
#     pickle.dump(clf_ends_upper_allophone, f)

# %% [markdown]
# ## Тестовая выборка

# %%
torch.manual_seed(42)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 5
NUM_DECODER_LAYERS = 5

g2p_model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in g2p_model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
        
# g2p_model = g2p_model.to(DEVICE)

g2p_model.load_state_dict(torch.load('phonetics/best_model.pth', map_location=torch.device('cpu')))
g2p_model = g2p_model.to(DEVICE)

# %%
with open('phonetics/clf_vowel_allophones_2.pkl', 'rb') as f:
    clf_vowel = pickle.load(f)
    
with open('phonetics/clf_vowel_allophones_nuclear_2.pkl', 'rb') as f:
    clf_nuclear = pickle.load(f)
    
with open('phonetics/clf_ends_upper_allophone.pkl', 'rb') as f:
    clf_ends_upper_allophone = pickle.load(f)
    
data_path = 'test/Test_Procody.xml' # TODO: ПОМЕНЯТЬ ДЛЯ РЕАЛЬНОГО ТЕСТА!!!!!!!!!!!!!!!!!111
tree = ET.parse(data_path)
root = tree.getroot()

# %%
dictitem = ['subpart_of_speech']
content = ['PunktEnd', 'PunktBeg', 'EmphEnd', 'EmphBeg']
phoneme_vowel = ['y', 'i', 'o', 'a', 'u', 'e']
phoneme_consonant = ['sh', 's', 't', 'l', 'k', 'r', 'n', 'j', 'b', 'z','v','g','h','d','ch','f','m','zh','c','p','sc']
grapheme_vowel = ["а", "е", "ё", "и", "о", "у", "ы", "э", "ю", "я"]
grapheme_consonant = ["б", "в", "г", "д", "ж", "з", "й", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ь"]
doubling = ['sh', 'sc', 'zh', 'ch']
end_phoneme_upper = ["H", "CH", "C", "SC", "h", 'ch', 'c', 'sc']

# %%
dct_feature_nuclear = {
                'phoneme_previous': [],# предыдущая фонема +
                'phoneme_current': [], # текущая фонема +
                'phoneme_next': [], # следующая фонема +
                'pos_phoneme_start': [], # позиция текущей фонемы от начала слова + 
                'pos_phoneme_end': [], # позиция текущей фонемы от конца слова +
                'flag_letter_previous': [], # ударность предыдущей фонемы +
                'flag_letter_current': [], # ударность текущей фонемы +
                'flag_letter_next': [], # ударность последующей фонемы +
                'subpart_of_speech_current': [], # часть речи текущего слова +
                'pos_word_start_sent': [], # позиция текущего слова в предложении от начала +
                'pos_word_end_sent': [], # позиция текущего слова в предложении от конца +
                'nucleus': [], # ударность слова +
                'pause': [],
                'pause_type': [],
                'letter_previous': [],
                'letter_current': [],
                'letter_next': [],
                'word': [],
                'phoneme_word': [],
                'num_word_before': [],
                'sentence_num': [],
                'phonem_place': [],
                }



for sent_idx in tqdm(range(0, len(root))):
    num_word_before = 0
    df_sent = pd.DataFrame()
    for tag_idx in range(len(root[sent_idx])):
        tag = root[sent_idx][tag_idx].tag
        attrib = root[sent_idx][tag_idx].attrib
        if tag == "word":
#             if "original" in attrib and attrib["original"] != '':
                try:
                    word_original = attrib["original"]
                except:
                    word_original = ""
                first_dictitem = True
                letter_list = []
                flag_letter_list = []

                flag_allo_cnt = True
                for child2 in root[sent_idx][tag_idx]:
                    if child2.tag == "dictitem":
                        if first_dictitem:                         
                            for feature_dictitem in dictitem:
                                if feature_dictitem in child2.keys():
                                    subpart_of_speech_current = int(child2.attrib[feature_dictitem])
                                else:
                                    subpart_of_speech_current = np.nan
                        first_dictitem = False
                    if child2.tag == "letter":
                        try:
                            letter = child2.attrib['char']
                        except:
                            letter = ''
                        try:
                            flag_letter = child2.attrib['flag']
                        except:
                            flag_letter = "0"
                        letter_list.append(letter)
                        flag_letter_list.append(flag_letter)
                    
#                 word_original = attrib["original"]
                if flag_allo_cnt:
                    idx_g_prev = 0
                phoneme_predict = inference(g2p_model, ''.join(letter_list)).replace(' \'', '\'').replace('s h', 'sh').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')[1:-1]
                pos_phoneme_start = 0
                for idx_p, p in enumerate(phoneme_predict):
                    if p in phoneme_vowel:
                        dct_feature_nuclear['word'].append(word_original)
                        dct_feature_nuclear['phoneme_word'].append("".join(phoneme_predict))
                        dct_feature_nuclear['pos_word_start_sent'].append(tag_idx)
                        dct_feature_nuclear['pos_word_end_sent'].append(len(root[sent_idx])-1-tag_idx)
                        dct_feature_nuclear['subpart_of_speech_current'].append(subpart_of_speech_current)
                        if "nucleus" in attrib:
                            if attrib["nucleus"] == "2":
                                dct_feature_nuclear['nucleus'].append(1)
                            else:
                                dct_feature_nuclear['nucleus'].append(0)
                        else:
                            dct_feature_nuclear['nucleus'].append(0)
                        if idx_p == 0:
                            dct_feature_nuclear['phoneme_previous'].append('<sos>')
                        else:                           
                            
                            dct_feature_nuclear['phoneme_previous'].append(text_transform[TGT_LANGUAGE](phoneme_predict[idx_p-1])[1].item())
                        if idx_p == len(phoneme_predict)-1:
                            dct_feature_nuclear['phoneme_next'].append('<eos>')
                        else:
                            dct_feature_nuclear['phoneme_next'].append(text_transform[TGT_LANGUAGE](phoneme_predict[idx_p+1])[1].item())
                        dct_feature_nuclear['phoneme_current'].append(text_transform[TGT_LANGUAGE](phoneme_predict[idx_p])[1].item())
                        dct_feature_nuclear['pos_phoneme_start'].append(pos_phoneme_start)
                        pos_phoneme_end = len(phoneme_predict) - 1 - idx_p
                        dct_feature_nuclear['pos_phoneme_end'].append(pos_phoneme_end)     
          
                        if p in phoneme_vowel:
                            idx_g = idx_p
                        else:
                            idx_g = idx_p
                            grapheme_predict = ''
                            while grapheme_predict not in phoneme_vowel and idx_g >= 0:
                                idx_g -= 1
                                if idx_g <= 0:
                                    break
                                if idx_g <= len(phoneme_predict) - 1:
                                    grapheme_predict = re.sub(r'\d+', r'', phoneme_predict[idx_g]).strip()
                        
                        idx_letter = idx_g if 'j' not in phoneme_predict[:idx_g] else idx_g-(phoneme_predict[:idx_g]).count('j')
                        for db in doubling:
                            if db in phoneme_predict[:idx_g]:
                                idx_letter = idx_letter - (phoneme_predict[:idx_g]).count(db)
              
                        if idx_g == 0:
                            dct_feature_nuclear['flag_letter_previous'].append("-1")
                            dct_feature_nuclear['letter_previous'].append('<sos>')
                        else:
                            dct_feature_nuclear['flag_letter_previous'].append(flag_letter_list[idx_letter-1])
                            dct_feature_nuclear['letter_previous'].append(text_transform[SRC_LANGUAGE](letter_list[idx_letter-1])[1].item())
                        if idx_g >= len(letter_list)-1:
                            dct_feature_nuclear['flag_letter_next'].append("-1")
                            dct_feature_nuclear['letter_next'].append('<eos>')
                        else:
                            dct_feature_nuclear['flag_letter_next'].append(flag_letter_list[idx_letter+1])
                            dct_feature_nuclear['letter_next'].append(text_transform[SRC_LANGUAGE](letter_list[idx_letter+1])[1].item())
                        dct_feature_nuclear['flag_letter_current'].append(flag_letter_list[idx_letter])
                        dct_feature_nuclear['letter_current'].append(text_transform[SRC_LANGUAGE](letter_list[idx_letter])[1].item())
    
                        
                        if root[sent_idx][tag_idx+2].tag == "pause":
                            dct_feature_nuclear['pause'].append(1)
                            dct_feature_nuclear['pause_type'].append(root[sent_idx][tag_idx+2].attrib["type"])
                        else:
                            dct_feature_nuclear['pause_type'].append('no')
                            dct_feature_nuclear['pause'].append(0)
                        dct_feature_nuclear['sentence_num'].append(sent_idx)
                        dct_feature_nuclear['num_word_before'].append(num_word_before)
                        dct_feature_nuclear['phonem_place'].append(idx_p)
                        idx_g_prev = idx_g
#                         df = pd.DataFrame(dct_feature_nuclear)
#                         display(df)
#                     break
#                         vowel_predict = clf_vowels(df)
#                         df['phonem_curent'] = vowel_predict
#                         df['phoneme_word'][idx_p] = vowel_predict
#                         df_sent.append(df)
#                         df['phoneme_word'] =                         
                    pos_phoneme_start += 1
                    
                num_word_before += 1
#         nuclear_vowel = clf_vowels(df_sent)
        

# %%
for k,v in dct_feature_nuclear.items():
    print(k, len(v))

# %%
df_test = pd.DataFrame(dct_feature_nuclear)
df_test.loc[df_test[df_test['phoneme_previous']=='<sos>'].index, 'phoneme_previous'] = vocab_transform[TGT_LANGUAGE]['<bos>']
df_test.loc[df_test[df_test['phoneme_previous']=='<eos>'].index, 'phoneme_previous'] = vocab_transform[TGT_LANGUAGE]['<eos>']
df_test.loc[df_test[df_test['phoneme_next']=='<sos>'].index, 'phoneme_next'] = vocab_transform[TGT_LANGUAGE]['<bos>']
df_test.loc[df_test[df_test['phoneme_next']=='<eos>'].index, 'phoneme_next'] = vocab_transform[TGT_LANGUAGE]['<eos>']

df_test.loc[df_test[df_test['letter_previous']=='<sos>'].index, 'letter_previous'] = vocab_transform[TGT_LANGUAGE]['<bos>']
df_test.loc[df_test[df_test['letter_previous']=='<eos>'].index, 'letter_previous'] = vocab_transform[TGT_LANGUAGE]['<eos>']
df_test.loc[df_test[df_test['letter_next']=='<sos>'].index, 'letter_next'] = vocab_transform[TGT_LANGUAGE]['<bos>']
df_test.loc[df_test[df_test['letter_next']=='<eos>'].index, 'letter_next'] = vocab_transform[TGT_LANGUAGE]['<eos>']

df_test = df_test.astype({'letter_current': int})
df_test

# %%
# sentence_num = df_test['sentence_num']
# df_test.drop('sentence_num', axis=1, inplace=True)

# %%
df_test.columns

# %%
sentence_num = df_test['sentence_num']
df_test.drop(['sentence_num', 'num_word_before', 'pos_phoneme_start'], inplace=True, axis=1)

# %%
# предскажим гласные 
vowels_pred = clf_vowel.predict(df_test)
vowels_pred_allophone = [vocab_transform[TGT_LANGUAGE].lookup_tokens([x])[0] for x in vowels_pred]
# предскажим ударение гласных
vowels_nuclear_pred = [str(x[0]) for x in clf_nuclear.predict(df_test)]

vowel_predict = [vowels_pred_allophone[idx]+vowels_nuclear_pred[idx] for idx in range(len(vowels_pred_allophone))]

# %%
df_predict = pd.DataFrame({'word': df_test['word'], 
                           'allophone_word': df_test['phoneme_word'], 
                           'allophone_place': df_test['phonem_place'], 
                           'vowel_predict': vowel_predict,
                           'sentence_num': sentence_num})


# %%
from copy import deepcopy
from more_itertools import unique_everseen
dct_finnaly = {'num_sent': [], 'word': [], 'allophone': []}
sent_num = np.unique(df_predict['sentence_num'])
for sent in sent_num:
    df_temp = deepcopy(df_predict[df_predict['sentence_num']==sent])
    df_temp.set_index('word', inplace=True)
    word_unique_sent = list(unique_everseen(df_temp.index))
    for word in word_unique_sent:
        idx_word = 0
        allophone = ''
        try:
            for _, row in df_temp.loc[word].iterrows():
                if idx_word == 0:
                    allophone = ' '.join(list(row['allophone_word'])).replace(' \'', '\'').replace('s h', 'sh').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')
                phoneme_place = row['allophone_place']
                vowel = row['vowel_predict']
                allophone[phoneme_place] = vowel
                idx_word += 1
        except:
            allophone = ' '.join(list(df_temp.loc[word]['allophone_word'])).replace(' \'', '\'').replace('s h', 'sh').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')
            phoneme_place = df_temp.loc[word]['allophone_place']
            vowel = df_temp.loc[word]['vowel_predict']
            allophone[phoneme_place] = vowel
        dct_finnaly['num_sent'].append(sent)
        dct_finnaly['word'].append(word)
        dct_finnaly['allophone'].append(allophone)           

# %%
df_finally = pd.DataFrame(dct_finnaly)
df_finally.head()

# %% [markdown]
# Предсказание конечных согласных букв

# %%
df_test = pd.DataFrame(dct_feature_nuclear)

# %%
idx_ph_list = []
for idx, phoneme in tqdm(df_test['phoneme_word'].items(), total=len(df_test)):
    end_ph = ' '.join(list(phoneme)).replace('s h', 'sh').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')[-1]
    if end_ph in end_phoneme_upper:
        idx_ph_list.append(idx)

# %%
df_test.loc[idx_ph_list]

# %%
dct_feature_vowels_upper = {
                'phoneme_previous': [],# предыдущая фонема +
                'phoneme_current': [], # текущая фонема +
                'phoneme_next': [], # следующая фонема +
                'flag_letter_previous': [], # ударность предыдущей фонемы +
                'flag_letter_current': [], # ударность текущей фонемы +
#                 'flag_letter_next': [], # ударность последующей фонемы +
                'subpart_of_speech_current': [], # часть речи текущего слова +
                'pos_word_start_sent': [], # позиция текущего слова в предложении от начала +
                'pos_word_end_sent': [], # позиция текущего слова в предложении от конца +
                'nucleus': [], # ударность слова +
                'pause': [],
                'pause_type': [],
                'letter_previous': [],
                'letter_current': [],
                'letter_next': [],
                'word': [],
                'phoneme_word': [],
                'num_word_before': [],
                'sentence_num': [],
                'phonem_place': [],
                }

    
for sent_idx in tqdm(np.unique(df_test.loc[idx_ph_list].sentence_num)):
    num_word_before = 0
    for tag_idx in range(len(root[sent_idx])):
        tag = root[sent_idx][tag_idx].tag
        attrib = root[sent_idx][tag_idx].attrib
        if tag == "word":
#             if "original" in attrib and attrib["original"] != '':
            try:
                word_original = attrib["original"]
            except:
                word_original = ""
#             if word_original == "":
#                 continue
            first_dictitem = True
            letter_list = []
            flag_letter_list = []     
            flag_allo_cnt = True
            for child2 in root[sent_idx][tag_idx]:
                if child2.tag == "dictitem":
                    if first_dictitem:                         
                        for feature_dictitem in dictitem:
                            if feature_dictitem in child2.keys():
                                subpart_of_speech_current = int(child2.attrib[feature_dictitem])
                            else:
                                subpart_of_speech_current = np.nan
                    first_dictitem = False
                if child2.tag == "letter":
                    letter = child2.attrib['char']
                    try:
                        flag_letter = child2.attrib['flag']
                    except:
                        flag_letter = "0"
                    letter_list.append(letter)
                    flag_letter_list.append(flag_letter)


            if flag_allo_cnt:
                idx_g_prev = 0
            if len(df_test[(df_test.sentence_num==sent_idx)&(df_test.word==word_original)])==0:
                continue
            
            phoneme_predict = ' '.join(list(df_test[(df_test.sentence_num==sent_idx)&(df_test.word==word_original)].phoneme_word)[0]).replace(' \'', '\'').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')

            try:
                try:
                    idx_next_word = list(df_test[(df_test.sentence_num==sent_idx)&(df_test.word==word_original)].index)[-1]+1
                    phoneme_next = ' '.join(list(df_test.loc[idx_next_word].phoneme_word)[0]).replace('s h', 'sh').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')
                    letter_next = list(df_test.loc[idx_next_word].word)[0]
                except: 
                    idx_next_word = list(df_test[(df_test.sentence_num==sent_idx)&(df_test.word==word_original)].index)[-1]+2
                    phoneme_next = ' '.join(list(df_test.loc[idx_next_word].phoneme_word)[0]).replace('s h', 'sh').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')
                    letter_next = list(df_test.loc[idx_next_word].word)[0]                
            except:
                phoneme_predict = [np.nan]
                letter_next = [np.nan]

            if phoneme_predict[-1] in end_phoneme_upper:     
                dct_feature_vowels_upper['word'].append(attrib['original'])
                dct_feature_vowels_upper['phoneme_word'].append("".join(phoneme_predict))
                dct_feature_vowels_upper['pos_word_start_sent'].append(tag_idx)
                dct_feature_vowels_upper['pos_word_end_sent'].append(len(root[sent_idx])-1-tag_idx)
                dct_feature_vowels_upper['subpart_of_speech_current'].append(subpart_of_speech_current)
                if "nucleus" in attrib:
                    if attrib["nucleus"] == "2":
                        dct_feature_vowels_upper['nucleus'].append(1)
                    else:
                        dct_feature_vowels_upper['nucleus'].append(0)
                else:
                    dct_feature_vowels_upper['nucleus'].append(0)                       
                dct_feature_vowels_upper['phoneme_previous'].append(text_transform[TGT_LANGUAGE](phoneme_predict[-2])[1].item())  
                dct_feature_vowels_upper['phoneme_next'].append(text_transform[TGT_LANGUAGE](phoneme_next)[1].item())      
                dct_feature_vowels_upper['phoneme_current'].append(text_transform[TGT_LANGUAGE](phoneme_predict[-1])[1].item())            
                pos_phoneme_end = len(phoneme_predict)    
                dct_feature_vowels_upper['flag_letter_previous'].append(flag_letter_list[-2])
                dct_feature_vowels_upper['letter_previous'].append(text_transform[SRC_LANGUAGE](letter_list[-2])[1].item())
                dct_feature_vowels_upper['letter_next'].append(letter_next)                                                    
                dct_feature_vowels_upper['flag_letter_current'].append(flag_letter_list[-1])
                dct_feature_vowels_upper['letter_current'].append(text_transform[SRC_LANGUAGE](letter_list[-1])[1].item())
                if root[sent_idx][tag_idx+2].tag == "pause":
                    dct_feature_vowels_upper['pause'].append(1)
                    dct_feature_vowels_upper['pause_type'].append(root[sent_idx][tag_idx+2].attrib["type"])
                else:
                    dct_feature_vowels_upper['pause_type'].append('no')
                    dct_feature_vowels_upper['pause'].append(0)
                dct_feature_vowels_upper['sentence_num'].append(sent_idx)
                dct_feature_vowels_upper['num_word_before'].append(num_word_before)
                dct_feature_vowels_upper['phonem_place'].append(pos_phoneme_end)
                    
            num_word_before += 1

# %%
for k,v in dct_feature_vowels_upper.items():
    print(k, len(v))

# %%
df_upper_lowels = pd.DataFrame(dct_feature_vowels_upper)
df_upper_lowels

# %% [markdown]
# Агрегация итогового результата

# %%
upper_lowels_predict = [x[0] for x in clf_ends_upper_allophone.predict(df_upper_lowels)]

# %%
df_ends = deepcopy(df_finally)
df_ends['end_upper_allophone'] = None

# %%
for idx_row, row in df_upper_lowels.iterrows():
    word = row['word']
    num_sent = row['sentence_num']
#     allophone = df_ends[(df_ends.word==word)&(df_ends.num_sent==num_sent)].allophone.item()
#     allophone_temp = deepcopy(allophone)
#     allophone_temp[-1] = upper_lowels_predict[idx_row]
    df_ends.loc[df_ends[(df_ends.word==word)&(df_ends.num_sent==num_sent)].index, 'end_upper_allophone'] = upper_lowels_predict[idx_row]

# %%
df_ends

# %%
torch.manual_seed(42)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 5
NUM_DECODER_LAYERS = 5

g2p_model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in g2p_model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
        
# g2p_model = g2p_model.to(DEVICE)

g2p_model.load_state_dict(torch.load('phonetics/best_model.pth', map_location=torch.device('cpu')))
g2p_model = g2p_model.to(DEVICE)

# %%
import re
lst_json = []
for sent_idx in tqdm(range(0, len(root))):
    dct_sent = {'words': []}
    for tag_idx in range(len(root[sent_idx])):
        tag = root[sent_idx][tag_idx].tag
        attrib = root[sent_idx][tag_idx].attrib
        if tag == "word":
#             if "original" in attrib:
            try:
                word = attrib["original"]
            except:
                word = ""
#             word = attrib['original']
            if len(df_ends[(df_ends.word==word)&(df_ends.num_sent==sent_idx)]) > 0:
                allophone = df_ends[(df_ends.word==word)&(df_ends.num_sent==sent_idx)].allophone.item()
                ends_upper_allophone = df_ends[(df_ends.word==word)&(df_ends.num_sent==sent_idx)].end_upper_allophone.item()
                if ends_upper_allophone is not None:
                    allophone[-1] = ends_upper_allophone                               
            else:
                allophone = inference(g2p_model, re.sub('[^а-яё]', '', word.lower())).replace(' \'', '\'').replace('s h', 'sh').replace('s c', 'sc').replace('z h', 'zh').replace('c h', 'ch').split(' ')[1:-1]
#         print(word)
            dct_word = {'content': word, 'allophones': allophone}
            dct_sent['words'].append(dct_word)               
    lst_json.append(dct_sent)

# %%
lst_json[:1]

# %%

result = {'words': []}
for sentence in lst_json:
    for wd in sentence['words']:
        word = {}
        word['content'] = wd['content']
        word['allophones'] = []

        it = 0
        while it < len(wd['allophones']):
            if it + 1 < len(wd['allophones']) and wd['allophones'][it + 1].isnumeric():
                word['allophones'].append(wd['allophones'][it] + wd['allophones'][it + 1])
                it += 2
            else:
                word['allophones'].append(wd['allophones'][it])
                it += 1

        result['words'].append(word)

result


# %%
import json
with open('test/test_data_2lab.json', 'w', encoding='utf-8') as f:
    json.dump([result], f, indent=4, ensure_ascii=False)

# %%
