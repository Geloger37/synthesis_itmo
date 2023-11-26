# %% [markdown]
# # Библиотеки

# %%
import numpy as np
import pandas as pd
import pickle
import xml.etree.ElementTree as ET
import json

from sklearn.metrics import mean_squared_error
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import ParameterGrid, KFold
from tqdm import tqdm
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

# # %% [markdown]
# # # Предобработка

# # %% [markdown]
# # Считывание и построения дерева XML файла

# # %%
# tree = ET.parse("data/combined.xml")
# root = tree.getroot()

# # %%
# list_of_all_allophones = ["<sos>", "<eos>", "a0", "a1", "a2", "a4", "b", "b'", "c", "ch", "C", "CH", "d", "d'", "e0", "e1", "e2", "e4", "f", "f'", "g", "g'", "h", "h'", "H", "i0", "i1", "i2", "i4", "j", "k", "k'", "l", "l'", "m", "m'", "n", "n'", "o0", "o1", "o2", "o4", "p", "p'", "r", "r'", "s", "sc", "sch", "sh", "SC", "s'", "t", "t'", "u0", "u1", "u2", "u4", "v", "v'", "y0", "y1", "y2", "y4", "z", "zh", "z'"]
# list_of_all_letters = ["<sos>", "<eos>", "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я", '-', '']
# dict_allophones = {value: idx for idx, value in enumerate(list_of_all_allophones)}
# dict_letter = {value: idx for idx, value in enumerate(list_of_all_letters)}
# pause_dict = {'no': -1, 'long': 1, 'weak': 2, 'spelling': 3, 'minimal': 4, 'x-long': 5}

# # %%
# dct_feature = {
#                 'word': [], #+  
# #                 'phoneme': [],
    
#                 'flag_letter_previous': [], # ударность предыдущей фонемы +
#                 'flag_letter_current': [], # ударность текущей фонемы +
#                 'flag_letter_next': [], # ударность последующей фонемы +
    
#                 'letter_previous': [], #+
#                 'letter_current': [], #+
#                 'letter_next': [], #+

#                 'nucleus': [], # ударность слова +
#                 'pause': [], #+ 
#                 'pause_type': [], #+   
    
#                 'power_stress': [],
#                 'type_stress': [],
    
#                 'subpart_of_speech': [], #+
#                 'pos_word_start_sent': [], # позиция текущего слова в предложении от начала +
#                 'pos_word_end_sent': [], # позиция текущего слова в предложении от конца +
    
#                 'allophone_previous': [], #+
#                 'allophone_current': [], #+
#                 'allophone_next': [], #+
#                 'sentence_num': [], #+
#                 'num_word_before': [], #+
#                 'pos_allophone_start': [], # позиция текущей фонемы от начала слова + 
#                 'pos_allophone_end': [], # позиция текущей фонемы от конца слова +
    
#                 'mfcc': []
#             }


# def catch(child2):
#     try:
#         return int(child2.attrib['flag'])
#     except:
#         return 0


# for sent_idx in tqdm(range(0, len(root))):
#     num_word_before = 0
#     for tag_idx in range(len(root[sent_idx])):
#         tag = root[sent_idx][tag_idx].tag
#         attrib = root[sent_idx][tag_idx].attrib
#         if tag == "word":
#             if "original" in attrib and attrib["original"] != '':
#                 word = attrib['original']
#                 first_dictitem = True
#                 allo_cnt = 0 
#                 letter_cnt = 0
#                 allophone_list = [child2.attrib["ph"] for child2 in root[sent_idx][tag_idx] if child2.tag == "allophone"]
#                 letter_list = []
#                 for child2 in root[sent_idx][tag_idx]:
#                     if child2.tag == "letter":
#                         try:
#                             letter_list.append(child2.attrib['char'])
#                         except:
#                             letter_list.append('')

#                 flag_letter_list = [catch(child2) for child2 in root[sent_idx][tag_idx] if child2.tag == "letter"]


#                 for idx, child2 in enumerate(root[sent_idx][tag_idx]):
                
#                     if child2.tag == 'allophone':
#                         if root[sent_idx][tag_idx][-1].tag == 'dictitem':
#                             subpart_of_speech = int(root[sent_idx][tag_idx][-1].attrib['subpart_of_speech'])
#                         else:
#                             subpart_of_speech = -1

#                         allophone_current = dict_allophones[child2.attrib["ph"]]
#                         mfcc = child2.attrib["mfcc"]

#                         if allo_cnt == 0:
#                             allophone_previous = dict_allophones['<sos>']
#                         else:
#                             allophone_previous = dict_allophones[allophone_list[allo_cnt-1]]

#                         if allo_cnt == len(allophone_list)-1:
#                             allophone_next = dict_allophones['<eos>']
#                         else:
#                             allophone_next = dict_allophones[allophone_list[allo_cnt+1]]

#                         allo_cnt += 1

# #                         if root[sent_idx][tag_idx][idx-1].tag == 'phoneme':
# #                              phoneme = root[sent_idx][tag_idx][idx-1].attrib["ph"]

#                         if root[sent_idx][tag_idx][idx-2].tag == 'letter':
#                             letter_current = dict_letter[root[sent_idx][tag_idx][idx-2].attrib["char"]]    
#                             flag_letter_current = catch(root[sent_idx][tag_idx][idx-2])
#                             if letter_cnt == 0:
#                                 letter_previous = dict_letter['<sos>']
#                                 flag_letter_previous = -1
#                             else:
#                                 letter_previous = dict_letter[letter_list[letter_cnt-1]]
#                                 flag_letter_previous = flag_letter_list[letter_cnt-1]

#                             if letter_cnt == len(letter_list)-1:
#                                 letter_next = dict_letter['<eos>']
#                                 flag_letter_previous = -1
#                             else:
#                                 letter_next = dict_letter[letter_list[letter_cnt+1]]
#                                 flag_letter_next = flag_letter_list[letter_cnt+1]
#                             letter_cnt += 1  
                            
#                         if root[sent_idx][tag_idx][idx+1].tag == 'stress':
#                             power = int(root[sent_idx][tag_idx][idx+1].attrib['power'])
#                             type_stress = int(root[sent_idx][tag_idx][idx+1].attrib['type'])
#                         else:
#                             power = -1
#                             type_stress = -1


#                         if "nucleus" in attrib:
#                             if attrib["nucleus"] == "2":
#                                 nucleus = 1
#                             else:
#                                 nucleus = 0
#                         else:
#                             nucleus = 0

#                         if root[sent_idx][tag_idx+2].tag == "pause":
#                             pause = 1
#                             pause_type = pause_dict[root[sent_idx][tag_idx+2].attrib["type"]]
#                         else:
#                             pause_type = pause_dict['no']
#                             pause = 0

#                         dct_feature['word'].append(word)
#                         dct_feature['mfcc'].append(mfcc)
#                         dct_feature['subpart_of_speech'].append(subpart_of_speech)
#                         dct_feature['allophone_current'].append(allophone_current)
#                         dct_feature['allophone_previous'].append(allophone_previous)
#                         dct_feature['allophone_next'].append(allophone_next)
#                         dct_feature['letter_current'].append(letter_current)
# #                         dct_feature['phoneme'].append(phoneme)
#                         dct_feature['letter_previous'].append(letter_previous)
#                         dct_feature['letter_next'].append(letter_next)
#                         dct_feature['flag_letter_current'].append(flag_letter_current)
#                         dct_feature['flag_letter_previous'].append(flag_letter_previous)
#                         dct_feature['flag_letter_next'].append(flag_letter_next)    
#                         dct_feature['pos_allophone_start'].append(allo_cnt)
#                         dct_feature['pos_allophone_end'].append(len(allophone_list) - 1 - allo_cnt)
#                         dct_feature['num_word_before'].append(num_word_before)
#                         dct_feature['pos_word_start_sent'].append(tag_idx)
#                         dct_feature['pos_word_end_sent'].append(len(root[sent_idx])-1-tag_idx)
#                         dct_feature['sentence_num'].append(sent_idx)
#                         dct_feature['nucleus'].append(nucleus)
#                         dct_feature['pause'].append(pause)
#                         dct_feature['pause_type'].append(pause_type)
#                         dct_feature['power_stress'].append(power)
#                         dct_feature['type_stress'].append(type_stress)
    
#         num_word_before += 1

# # %%
# df = pd.DataFrame(dct_feature)
# print(df)

# # %% [markdown]
# # Нормализация данных

# # %%
# df_t = deepcopy(df)
# df_t.drop(['word', 'mfcc'], axis=1, inplace=True)

# scaler = StandardScaler()
# X = scaler.fit_transform(df_t)
# df_scaler = pd.DataFrame(X, columns=df_t.columns)
# df_scaler['mfcc'] = df['mfcc']

# # %%
# # Save the scaler
# with open('acoustics/scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)

# # %%
# X_train, X_test = train_test_split(df_scaler, test_size=0.1, random_state=42, shuffle=False)
# X_test.reset_index(inplace=True)
# X_test.drop('index', inplace=True, axis=1)
# columns = list(X_train.columns)
# columns.remove('mfcc')

# # %% [markdown]
# # # Обучение

# # %%
# def catboost_GridSearchCV(X, y, params, n_splits=2):
#     ps = {'rmse': 100,
#           'param': []
#     }
    
#     predict=None
    
#     for prms in tqdm(list(ParameterGrid(params)), ascii=True, desc='Params Tuning:'):
#         print(prms)
                          
#         rmse = cross_val(X, y, prms, n_splits=n_splits)

#         if rmse<ps['rmse']:
#             ps['rmse'] = rmse
#             ps['param'] = prms
#     print(f"RMSE: {str(ps['rmse'])}")
#     print(f"Params: {str(ps['param'])}")
    
#     return ps['param']

# # %%
# def cross_val(X, y, param,n_splits=3):
#     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
#     rmse_list = []
#     predict = None
    
#     for tr_ind, val_ind in kfold.split(X, y):
# #         print(tr_ind)
#         X_train = X.loc[tr_ind]
#         y_train = y.loc[tr_ind]
        
#         X_valid = X.loc[val_ind]
#         y_valid = y.loc[val_ind]
        
#         clf = CatBoostRegressor(iterations=1000,
#                                 loss_function = 'MultiRMSE',
#                                 depth=param['depth'],
#                                 l2_leaf_reg = param['l2_leaf_reg'],
#                                 eval_metric = 'MultiRMSE',
#                                 use_best_model=True,
#                                 task_type="GPU",
#                                 learning_rate=1,
#                                 metric_period=100
#         )
        
#         clf.fit(X_train, 
#                 y_train,
# #                 cat_features=None,
#                 eval_set=(X_valid, y_valid)
#         )
        
#         y_pred = clf.predict(X_valid)
#         rmse = mean_squared_error(y_valid, y_pred)
#         rmse_list.append(rmse)
#         break
#     return sum(rmse_list)/n_splits

# # %%
# X = deepcopy(df_scaler)
# y = pd.DataFrame([np.array(x.split('|')[1:-1], dtype=float) for x in X['mfcc']])
# X.drop('mfcc', axis=1, inplace=True)

# # %%
# reg = np.logspace(-20, -19, 1)
# reg = np.append(reg, [3, 10])
# params = {'depth': [4, 6, 10],
#           'l2_leaf_reg': reg
# }

# param = catboost_GridSearchCV(X, y, params)
# print(param)

# # %%
# x_train = deepcopy(X_train)
# y_train = pd.DataFrame(np.array(x.split('|')[1:-1], dtype=float) for x in x_train['mfcc'])
# x_train.drop('mfcc', axis=1, inplace=True)

# x_test = deepcopy(X_test)
# y_test = pd.DataFrame(np.array(x.split('|')[1:-1], dtype=float) for x in x_test['mfcc'])
# x_test.drop('mfcc', axis=1, inplace=True)

# # %%
# params = {'learning_rate': 0.3, 
#           'depth': 6, 
#           'l2_leaf_reg': 10.0, 
#           'loss_function': 'MultiRMSE', 
#           'eval_metric': 'MultiRMSE', 
#           'task_type': 'GPU', 
#           'iterations': 2000,
#            'use_best_model': True,
#           'od_type': 'Iter', 
#           'boosting_type': 'Plain', 
#           'bootstrap_type': 'Bernoulli', 
#           'allow_const_label': True, 
#           'metric_period': 50
#          }
# model_reg = CatBoostRegressor(**params)
# model_reg.fit(x_train, y_train,eval_set=(x_test, y_test))

# # %%
# y_predict = model_reg.predict(x_test)

# # %%
# cos_dist = []
# cos_sim = []
# for x, y in tqdm(zip(y_predict, np.array(y_test)), total=len(y_test)):
#     cos_dist.append(cosine_distances([x], [y]))
#     cos_sim.append(cosine_similarity([x], [y]))
# print(f'cosine_distances: {np.mean(cos_dist)}')
# print(f'cos_similarity: {np.mean(cos_sim)}')

# # %%
# with open('acoustics/catboost_regressor_best.pkl', 'wb') as f:
#     pickle.dump(model_reg, f)

# %% [markdown]
# # Тестирование


# %%
with open('acoustics/catboost_regressor_best.pkl', 'rb') as f:
    model = pickle.load(f)

with open('acoustics/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# %%
list_of_all_allophones = ["<sos>", "<eos>", "a0", "a1", "a2", "a4", "b", "b'", "c", "ch", "C", "CH", "d", "d'", "e0", "e1", "e2", "e4", "f", "f'", "g", "g'", "h", "h'", "H", "i0", "i1", "i2", "i4", "j", "k", "k'", "l", "l'", "m", "m'", "n", "n'", "o0", "o1", "o2", "o4", "p", "p'", "r", "r'", "s", "sc", "sch", "sh", "SC", "s'", "t", "t'", "u0", "u1", "u2", "u4", "v", "v'", "y0", "y1", "y2", "y4", "z", "zh", "z'"]
list_of_all_letters = ["<sos>", "<eos>", "а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я", '-', '']
dict_allophones = {value: idx for idx, value in enumerate(list_of_all_allophones)}
dict_letter = {value: idx for idx, value in enumerate(list_of_all_letters)}
pause_dict = {'no': -1, 'long': 1, 'weak': 2, 'spelling': 3, 'minimal': 4, 'x-long': 5}


# %%
tree_test = ET.parse("data/Ten_nad_Innsmutom.xml") # ПОМЕНЯТЬ ДЛЯ ТЕСТОВОГО ФАЙЛА
root_test = tree_test.getroot()

# %%
def catch(child2):
    try:
        return int(child2.attrib['flag'])
    except:
        return 0

# %%
lst_json = []
for sent_idx in tqdm(range(0, len(root_test))):
    num_word_before = 0
    dct_sent = {'words': []}
    for tag_idx in range(len(root_test[sent_idx])):
        tag = root_test[sent_idx][tag_idx].tag
        attrib = root_test[sent_idx][tag_idx].attrib
#         if "original" in attrib and attrib["original"] != '':
#         if "original" in attrib:
        if tag == "word":

            try:
                word = attrib["original"]
            except:
                word = ''
            mfccs = []

            first_dictitem = True
            allo_cnt = 0 
            letter_cnt = 0
            allophone_list = [child2.attrib["ph"] for child2 in root_test[sent_idx][tag_idx] if child2.tag == "allophone"]
            letter_list = []
            for child2 in root_test[sent_idx][tag_idx]:
                if child2.tag == "letter":
                    try:
                        letter_list.append(child2.attrib['char'])
                    except:
                        letter_list.append('')
            flag_letter_list = [catch(child2) for child2 in root_test[sent_idx][tag_idx] if child2.tag == "letter"]       

            for idx, child2 in enumerate(root_test[sent_idx][tag_idx]):

                if child2.tag == 'allophone':
                    dct_feature = {
                                'flag_letter_previous': [], # ударность предыдущей фонемы +
                                'flag_letter_current': [], # ударность текущей фонемы +
                                'flag_letter_next': [], # ударность последующей фонемы +
                                'letter_previous': [], #+
                                'letter_current': [], #+
                                'letter_next': [], #+
                                'nucleus': [], # ударность слова +
                                'pause': [], #+ 
                                'pause_type': [], #+   
                                'power_stress': [],
                                'type_stress': [],
                                'subpart_of_speech': [], #+
                                'pos_word_start_sent': [], # позиция текущего слова в предложении от начала +
                                'pos_word_end_sent': [], # позиция текущего слова в предложении от конца +
                                'allophone_previous': [], #+
                                'allophone_current': [], #+
                                'allophone_next': [], #+
                                'sentence_num': [], #+
                                'num_word_before': [], #+
                                'pos_allophone_start': [], # позиция текущей фонемы от начала слова + 
                                'pos_allophone_end': [], # позиция текущей фонемы от конца слова + 
                                }

                    if root_test[sent_idx][tag_idx][-1].tag == 'dictitem':
                        subpart_of_speech = int(root_test[sent_idx][tag_idx][-1].attrib['subpart_of_speech'])
                    else:
                        subpart_of_speech = -1

                    allophone_current = dict_allophones[child2.attrib["ph"]]

                    if allo_cnt == 0:
                        allophone_previous = dict_allophones['<sos>']
                    else:
                        allophone_previous = dict_allophones[allophone_list[allo_cnt-1]]

                    if allo_cnt == len(allophone_list)-1:
                        allophone_next = dict_allophones['<eos>']
                    else:
                        allophone_next = dict_allophones[allophone_list[allo_cnt+1]]

                    allo_cnt += 1
                    if root_test[sent_idx][tag_idx][idx-2].tag == 'letter':
                        letter_current = dict_letter[root_test[sent_idx][tag_idx][idx-2].attrib["char"]]    
                        flag_letter_current = catch(root_test[sent_idx][tag_idx][idx-2])
                        if letter_cnt == 0:
                            letter_previous = dict_letter['<sos>']
                            flag_letter_previous = -1
                        else:
                            letter_previous = dict_letter[letter_list[letter_cnt-1]]
                            flag_letter_previous = flag_letter_list[letter_cnt-1]

                        if letter_cnt == len(letter_list)-1:
                            letter_next = dict_letter['<eos>']
                            flag_letter_previous = -1
                        else:
                            letter_next = dict_letter[letter_list[letter_cnt+1]]
                            flag_letter_next = flag_letter_list[letter_cnt+1]
                        letter_cnt += 1  

                    if root_test[sent_idx][tag_idx][idx+1].tag == 'stress':
                        power = int(root_test[sent_idx][tag_idx][idx+1].attrib['power'])
                        type_stress = int(root_test[sent_idx][tag_idx][idx+1].attrib['type'])
                    else:
                        power = -1
                        type_stress = -1
                    if "nucleus" in attrib:
                        if attrib["nucleus"] == "2":
                            nucleus = 1
                        else:
                            nucleus = 0
                    else:
                        nucleus = 0

                    if root_test[sent_idx][tag_idx+2].tag == "pause":
                        pause = 1
                        pause_type = pause_dict[root_test[sent_idx][tag_idx+2].attrib["type"]]
                    else:
                        pause_type = pause_dict['no']
                        pause = 0

    #                 dct_feature['word'].append(word)
                    dct_feature['subpart_of_speech'].append(subpart_of_speech)
                    dct_feature['allophone_current'].append(allophone_current)
                    dct_feature['allophone_previous'].append(allophone_previous)
                    dct_feature['allophone_next'].append(allophone_next)
                    dct_feature['letter_current'].append(letter_current)
    #                         dct_feature['phoneme'].append(phoneme)
                    dct_feature['letter_previous'].append(letter_previous)
                    dct_feature['letter_next'].append(letter_next)
                    dct_feature['flag_letter_current'].append(flag_letter_current)
                    dct_feature['flag_letter_previous'].append(flag_letter_previous)
                    dct_feature['flag_letter_next'].append(flag_letter_next)    
                    dct_feature['pos_allophone_start'].append(allo_cnt)
                    dct_feature['pos_allophone_end'].append(len(allophone_list) - 1 - allo_cnt)
                    dct_feature['num_word_before'].append(num_word_before)
                    dct_feature['pos_word_start_sent'].append(tag_idx)
                    dct_feature['pos_word_end_sent'].append(len(root_test[sent_idx])-1-tag_idx)
                    dct_feature['sentence_num'].append(sent_idx)
                    dct_feature['nucleus'].append(nucleus)
                    dct_feature['pause'].append(pause)
                    dct_feature['pause_type'].append(pause_type)
                    dct_feature['power_stress'].append(power)
                    dct_feature['type_stress'].append(type_stress)

                    df = pd.DataFrame(dct_feature)
                    X = scaler.transform(df)
                    mfcc = model.predict(X)
                    mfccs.append(mfcc[0].tolist())

            dct_word = {'content': word, 'mfcc': mfccs}
            dct_sent['words'].append(dct_word)

        num_word_before += 1
    lst_json.append(dct_sent) 

# %%
result = {'words': []}
for sentence in lst_json:
    for wd in sentence['words']:
        word = {}
        word['content'] = wd['content']
        word['mfcc'] = wd['mfcc']

        result['words'].append(word)

print(result)


# %%
with open('test/test_data_3lab.json', 'w', encoding='utf-8') as f:
    json.dump([result], f, indent=4, ensure_ascii=False)
