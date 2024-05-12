"""
Created on Sat Feb 29 21:26:17 2020

@author: Roghi
"""

from nltk import bigrams, trigrams
from nltk.util import ngrams
from collections import Counter

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE,Laplace,Lidstone

import glob
import re
import numpy as np
from random import shuffle
from bs4 import BeautifulSoup

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import dill as pickle

#------------------------------------------------------------------------------
from parsivar import Normalizer
from parsivar import Tokenizer
from parsivar import FindStems

my_stemmer = FindStems()
my_normalizer = Normalizer()
my_tokenizer = Tokenizer()

#------------------------------------------------------------------------------
def denoise_text(text):    
    # removing the html strips
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    pattern=r'[a-zA-Z]'
    text=re.sub(pattern,' ',text)
    
    # string.punctuation is '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    notAllowedPunctuations = '"#$%&\'()*+/;<=>?@[\\]^`{|}~'    
    table = str.maketrans(notAllowedPunctuations,str.ljust(' ', len(notAllowedPunctuations)),'')
    text = text.translate(table) 
    
    return text
#------------------------------------------------------------------------------
def read_train_test_data(files_path):
    texts = glob.glob(files_path)
    texts.sort()
    #shuffle(culture_texts)
    #print(list(texts))
    num_of_train_texts = int(0.8 * len(texts))
    print("number_of_train_texts:" , num_of_train_texts)
    print("number_of_test_texts:" , len(texts) - num_of_train_texts)
    
    train_texts = ""
    test_text_list = []
    for k in range(len(texts)):
        text_file = texts[k]    
        fb = open(text_file, "r",encoding="utf-8")
        if (k <= num_of_train_texts):
            train_texts += (" " + denoise_text(fb.read()))
        else:
            test_text_list.append(denoise_text(fb.read()))
           
    return train_texts, test_text_list        
#------------------------------------------------------------------------------
def tokenized_texts(train_texts):
        
    train_sents = my_tokenizer.tokenize_sentences(my_normalizer.normalize(train_texts))

    print(len(train_sents))
    print("sample sentence 2 of train_culture_sents:\n" + train_sents[2])

    tokenized_word_train_text = []
    plain_train_text = []
    for s in train_sents:
        words = my_tokenizer.tokenize_words(s)
        words_stemming=[]
        for w in words:
            words_stemming.append(my_stemmer.convert_to_stem(w))
        
        tokenized_word_train_text.append(words_stemming)
        
        plain_train_text.append(s)
        
    #print(len(tokenized_word_train_text[2]))
    #print("sample tokenized sentence 2 of train_culture_sents:\n" , tokenized_word_train_text[2]) 
    return tokenized_word_train_text, plain_train_text
#------------------------------------------------------------------------------    
def create_lm_ngram(n, tokenized_word_train_text):

    train_ngrams, vocab_ngrams = padded_everygram_pipeline(n, tokenized_word_train_text)
        
    #lm_ngram = MLE(n)
    lm_ngram = Laplace(n)
    #lm_ngram = Lidstone(1)
    lm_ngram.fit(train_ngrams, vocab_ngrams)
    print(lm_ngram.vocab)
    return lm_ngram
#------------------------------------------------------------------------------    
def evaluate_lm(lm_type, n, test_text_list, label_indx, lm_cul, lm_fin, lm_pol, lm_soc, lm_spr, lm_tec):
    y_test = []
    y_pred = []
    perp_pred = []
    for test_text in test_text_list:
        test_sents = my_tokenizer.tokenize_sentences(my_normalizer.normalize(test_text))

        tokenized_test_text = []
        for s in test_sents:
            if (lm_type == 'word'):
                words = my_tokenizer.tokenize_words(s)
                words_stemming=[]
                for w in words:
                    words_stemming.append(my_stemmer.convert_to_stem(w))
                tokenized_test_text.append(words_stemming)
            elif (lm_type == 'char'):
                tokenized_test_text.append(s)
        #print(list(tokenized_test_text))        
        
        test_data, _ = padded_everygram_pipeline(n, tokenized_test_text)        
        perplexity_test = []
        perplexity_test_cul = []        
        for test in test_data:
            try:
                perplexity_test_cul.append(lm_cul.perplexity(test))
            except:
                perplexity_test_cul.append(100000)
        perplexity_test.append(min(perplexity_test_cul))
        #--------------------------------------------------   
        test_data, _ = padded_everygram_pipeline(n, tokenized_test_text)
        perplexity_test_fin = []        
        for test in test_data:
            try:    
                perplexity_test_fin.append(lm_fin.perplexity(test))
            except:
                perplexity_test_fin.append(100000)
        perplexity_test.append(min(perplexity_test_fin))
        #--------------------------------------------------    
        test_data, _ = padded_everygram_pipeline(n, tokenized_test_text)        
        perplexity_test_pol = []        
        for test in test_data:
            try:
                perplexity_test_pol.append(lm_pol.perplexity(test))
            except:
                perplexity_test_pol.append(100000)
        perplexity_test.append(min(perplexity_test_pol))                
        #--------------------------------------------------   
        test_data, _ = padded_everygram_pipeline(n, tokenized_test_text)        
        perplexity_test_soc = []        
        for test in test_data:
            try:
                perplexity_test_soc.append(lm_soc.perplexity(test))
            except:
                perplexity_test_soc.append(100000)
        perplexity_test.append(min(perplexity_test_soc))                                
        #--------------------------------------------------  
        test_data, _ = padded_everygram_pipeline(n, tokenized_test_text)        
        perplexity_test_spr = []        
        for test in test_data:
            try:
                perplexity_test_spr.append(lm_spr.perplexity(test))
            except:
                perplexity_test_spr.append(100000)
        perplexity_test.append(min(perplexity_test_spr))                                
        #--------------------------------------------------   
        test_data, _ = padded_everygram_pipeline(n, tokenized_test_text)        
        perplexity_test_tec = []        
        for test in test_data:
            try:
                perplexity_test_tec.append(lm_tec.perplexity(test))
            except:
                perplexity_test_tec.append(100000)
        perplexity_test.append(min(perplexity_test_tec))                                                
        #--------------------------------------------------    
        #print(perplexity_test)
        pred_indx = perplexity_test.index(min(perplexity_test))
        #print(pred_indx)
        y_pred.append(pred_indx)
        #print(y_pred)
        perp_pred.append(min(perplexity_test))
        y_test.append(label_indx)
        #print(y_test)
        
    print(f1_score(y_test, y_pred, average="micro"))
    print(precision_score(y_test, y_pred, average="micro"))
    print(recall_score(y_test, y_pred, average="micro"))  
    return y_test,y_pred, perp_pred
#------------------------------------------------------------------------------ 
def perplexity_all_test_texts(test_text_list, lm_unigram_word, lm_bigram_word, lm_unigram_char, lm_bigram_char):
    test_text = ""
    for txt in test_text_list:
        test_text += (" " + txt)
        
    test_txt_sents = my_tokenizer.tokenize_sentences(my_normalizer.normalize(test_text))

    tokenized_test_txt = []
    plain_test_txt = []
    for s in test_txt_sents:
        tokenized_test_txt.append(my_tokenizer.tokenize_words(s))
        plain_test_txt.append(s)
    #--------------------------------------------------------
    test_data, _ = padded_everygram_pipeline(1, tokenized_test_txt)
    perp_test = []
    for test in test_data:
        try:
            perp_test.append(lm_unigram_word.perplexity(test))
        except:
            pass
    
    print("lm_unigram_word--> min(perplexity) : " , np.min(perp_test))
    #--------------------------------------------------------
    test_data, _ = padded_everygram_pipeline(2, tokenized_test_txt)
    perp_test = []
    for test in test_data:
        try:
            perp_test.append(lm_bigram_word.perplexity(test))
        except:
            pass
    
    print("lm_bigram_word--> min(perplexity) : " , np.min(perp_test))
    #--------------------------------------------------------
    test_data, _ = padded_everygram_pipeline(1, plain_test_txt)
    perp_test = []
    for test in test_data:
        try:
            perp_test.append(lm_unigram_char.perplexity(test))
        except:
            pass
    
    print("lm_unigram_char--> min(perplexity) : " , np.min(perp_test))
    #--------------------------------------------------------
    test_data, _ = padded_everygram_pipeline(2, plain_test_txt)
    perp_test = []
    for test in test_data:
        try:
            perp_test.append(lm_bigram_char.perplexity(test))
        except:
            pass
    
    print("lm_bigram_char--> min(perplexity) : " , np.min(perp_test))
    return
#------------------------------------------------------------------------------    
def create_lms_for_class(train_texts, class_name):
    tokenized_word_train_text, plain_train_text = tokenized_texts(train_texts)

    lm_unigram_word = create_lm_ngram(1, tokenized_word_train_text)
    lm_bigram_word = create_lm_ngram(2, tokenized_word_train_text)
    
    lm_unigram_char = create_lm_ngram(1, plain_train_text)
    lm_bigram_char = create_lm_ngram(2, plain_train_text)
    
    with open ('lm_unigram_word_'+class_name+'.pkl','wb') as fout:
        pickle.dump(lm_unigram_word,fout)
        
    with open ('lm_bigram_word_'+class_name+'.pkl','wb') as fout:
        pickle.dump(lm_bigram_word,fout)
        
    with open ('lm_unigram_char_'+class_name+'.pkl','wb') as fout:
        pickle.dump(lm_unigram_char,fout)
        
    with open ('lm_bigram_char_'+class_name+'.pkl','wb') as fout:
        pickle.dump(lm_bigram_char,fout)

    return
#------------------------------------------------------------------------------    
def load_lms_for_class(class_name):
        with open ('lm_unigram_word_'+class_name+'.pkl','rb') as fin:
            lm_unigram_word = pickle.load(fin)
            
        with open ('lm_bigram_word_'+class_name+'.pkl','rb') as fin:
            lm_bigram_word = pickle.load(fin)
            
        with open ('lm_unigram_char_'+class_name+'.pkl','rb') as fin:
            lm_unigram_char = pickle.load(fin)
            
        with open ('lm_bigram_char_'+class_name+'.pkl','rb') as fin:
            lm_bigram_char = pickle.load(fin)
            
        return lm_unigram_word, lm_bigram_word, lm_unigram_char, lm_bigram_char   
#------------------------------------------------------------------------------            
train_cul_texts, test_cul_text_list = read_train_test_data("E:\\Projects\\CA-1\\train\\culture\\*.txt")
print("*********")
#create_lms_for_class(train_cul_texts,'cul')

lm_unigram_word_cul, lm_bigram_word_cul, lm_unigram_char_cul, lm_bigram_char_cul = load_lms_for_class('cul')

#perplexity_all_test_texts(test_cul_text_list, lm_unigram_word_cul, lm_bigram_word_cul, lm_unigram_char_cul, lm_bigram_char_cul)

#******************************************************************************
train_fin_texts, test_fin_text_list = read_train_test_data("E:\\Projects\\CA-1\\train\\finance\\*.txt")
print("*********")
#create_lms_for_class(train_fin_texts,'fin')

lm_unigram_word_fin, lm_bigram_word_fin, lm_unigram_char_fin, lm_bigram_char_fin = load_lms_for_class('fin')

#perplexity_all_test_texts(test_fin_text_list, lm_unigram_word_fin, lm_bigram_word_fin, lm_unigram_char_fin, lm_bigram_char_fin)

#******************************************************************************
train_pol_texts, test_pol_text_list = read_train_test_data("E:\\Projects\\CA-1\\train\\politic\\*.txt")
print("*********")
#create_lms_for_class(train_pol_texts,'pol')

lm_unigram_word_pol, lm_bigram_word_pol, lm_unigram_char_pol, lm_bigram_char_pol = load_lms_for_class('pol')

#perplexity_all_test_texts(test_pol_text_list, lm_unigram_word_pol, lm_bigram_word_pol, lm_unigram_char_pol, lm_bigram_char_pol)

#******************************************************************************
train_soc_texts, test_soc_text_list = read_train_test_data("E:\\Projects\\CA-1\\train\\social\\*.txt")
print("*********")
#create_lms_for_class(train_soc_texts,'soc')

lm_unigram_word_soc, lm_bigram_word_soc, lm_unigram_char_soc, lm_bigram_char_soc = load_lms_for_class('soc')

#perplexity_all_test_texts(test_soc_text_list, lm_unigram_word_soc, lm_bigram_word_soc, lm_unigram_char_soc, lm_bigram_char_soc)

#******************************************************************************
train_spr_texts, test_spr_text_list = read_train_test_data("E:\\Projects\\CA-1\\train\\sport\\*.txt")
print("*********")
#create_lms_for_class(train_spr_texts,'spr')

lm_unigram_word_spr, lm_bigram_word_spr, lm_unigram_char_spr, lm_bigram_char_spr = load_lms_for_class('spr')

#perplexity_all_test_texts(test_spr_text_list, lm_unigram_word_spr, lm_bigram_word_spr, lm_unigram_char_spr, lm_bigram_char_spr)

#******************************************************************************
train_tec_texts, test_tec_text_list = read_train_test_data("E:\\Projects\\CA-1\\train\\technology\\*.txt")
print("*********")
#create_lms_for_class(train_tec_texts,'tec')

lm_unigram_word_tec, lm_bigram_word_tec, lm_unigram_char_tec, lm_bigram_char_tec = load_lms_for_class('tec')

#perplexity_all_test_texts(test_tec_text_list, lm_unigram_word_tec, lm_bigram_word_tec, lm_unigram_char_tec, lm_bigram_char_tec)

#******************************************************************************
#print(test_cul_text_list[1:2])
#test_cul_text_list=test_cul_text_list[1:2]
#test_cul_text_list=[['در سال 1385 با سفر محمود احمدی نژاد به مالزی و در خواست ایرانیان و دانشجویان مقیم مالزی .']]

print("*********------------------------")
y_test,y_pred, prep_pred = evaluate_lm('char', 1, test_tec_text_list, 5, 
                                            lm_unigram_char_cul, 
                                            lm_unigram_char_fin, 
                                            lm_unigram_char_pol, 
                                            lm_unigram_char_soc, 
                                            lm_unigram_char_spr, 
                                            lm_unigram_char_tec)
print("y_test:",y_test)
print("y_pred:",y_pred)

#print("prep_pred:",prep_pred)

#******************************************************************************

