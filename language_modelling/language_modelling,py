# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import gzip
import zipfile
import string
import re
import nltk
from nltk import word_tokenize
from nltk import ngrams
from collections import Counter
from matplotlib import pyplot


nltk.download('punkt')

with gzip.open('train.corpus.gz', 'rt') as f:
    raw_txt = f.read()

def first_question(raw_txt):
    #1.a)
    tokens = raw_txt.split()
    num_tokens = len(tokens)
    print("1.a) Number of running words (tokens) including the punctuation marks : ", num_tokens)
    #1.b)
    lines = raw_txt.split("\n")
    num_lines = len(lines)
    print("1.b) Number of sentences in the corpus : ", num_lines)
    #1.c)
    sentence_lengths = [len(x) for x in lines]
    avg_sentence_length = np.mean(sentence_lengths)
    med_sentence_length = np.median(sentence_lengths)
    sentence_length_count = Counter(sentence_lengths)
    print("1.c)Mean sentence length : ", avg_sentence_length)
    print("Median sentence length : ", med_sentence_length)
    sentence_length_count = dict(sentence_length_count)
    lists = sorted(sentence_length_count.items())
    pyplot.yscale('log')
    x, y = zip(*lists)

    #print("Sentence Length count ", sentence_length_count)
    pyplot.plot(x, y)
    pyplot.title("Sentence length count plot")
    pyplot.xlabel("Sentence Length")
    pyplot.ylabel("Count")
    pyplot.yscale('log')

    fig = pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('test2png.png', dpi=100)

    pyplot.show()

    return raw_txt

raw_txt = first_question(raw_txt)

def part_d(txt, words_list):
  
  print("Test ", words_list[:10])
  
  num_words_corpus = len(words_list)
  
  vocab_set = set(line.strip() for line in open('vocabulary'))
  print("Num of words in vocab : ", len(vocab_set))
  txt_vocab_list = [x if x in vocab_set else '<unk>' for x in words_list]
 
  print("TEST SIZE ", len(txt_vocab_list))
  
  #print("TTXT : ", txt[:650])
  #print("Marked text ", txt_vocab_list[:80])
  trigrams_vocab = ngrams(txt_vocab_list, 3)
  
 
  #print("Num of trigrams based on vocabulary: ", sum(1 for x in trigrams_vocab))
  
  
  cnt = 1
  for x in trigrams_vocab:
    cnt += 1
    if cnt > 10 :
      break
    print(x)
  
  
  trigrams_count = Counter(trigrams_vocab)
  print("Num of trigrams based on vocabulary : ", sum(trigrams_count.values()), sum(1 for x,y in trigrams_count.items()))

  
  
  top_10_trigrams = trigrams_count.most_common(10)
  print("Top 10 trigrams based on vocabulary: ", top_10_trigrams)
  
  
  print("Number of words in corpus : ", num_words_corpus)
  
  num_out_of_vocab = np.count_nonzero(np.array(txt_vocab_list) == "<unk>")
  
  print("Number of words that are out of vocabulary : ", num_out_of_vocab)
  
  out_of_vocab_rate = (num_out_of_vocab / num_words_corpus) * 100.0
  
  print("Out of Vocabulary rate : ", out_of_vocab_rate)
  
  return txt_vocab_list, trigrams_count

def second_question(raw_txt):
    print("Removing punctuations before finding words...")
    txt_no_punc = raw_txt.translate(str.maketrans('', '', string.punctuation))
    #words_list = txt_no_punc.split()
    words_list = re.findall(r'\w+', txt_no_punc)
    
    
    
    num_unique_words = len(set(words_list))
    words_count = Counter(words_list)
    
    assert num_unique_words == len(set(words_list))
    top_10_words = words_count.most_common(10)
    
    print("2.a) Most common words : ", top_10_words)
    
    
    sentences = txt_no_punc.split("\n")
    sentences_marked = ['<s> ' + s + ' </s>' for s in sentences] 
    print(sentences_marked[:5])
    txt_no_punc = ' '.join(sentences_marked)
    
    print("Test passed")
    
    
    words_list = txt_no_punc.split()
    print("Marked text ", len(words_list), words_list[:80])
    #tokens = nltk.word_tokenize(txt_no_punc)
    trigrams = ngrams(words_list, 3)
    
    #temp = sum(1 for x in trigrams)
    
    print("2.b")
    
    trigrams_count = Counter(trigrams)
    top_10_trigrams = trigrams_count.most_common(10)
    
    print("Num of trigrams : ", sum(trigrams_count.values()), sum(1 for x,y in trigrams_count.items()))
    print("2.c")
    print("Top 10 trigrams: ", top_10_trigrams)
    
    #print(list(sorted(dict(trigrams_count).values(), reverse=True))[:50])
    
    
    trigrams_count = Counter(dict(trigrams_count).values())
    print("Check ", dict(trigrams_count)[3])
    
    lists = dict(trigrams_count).items()
    pyplot.yscale('log')
    x, y = zip(*lists)

    #print("Sentence Length count ", sentence_length_count)
    pyplot.plot(x, y)
    pyplot.title("Trigrams count plot")
    pyplot.xlabel("Trigrams")
    pyplot.ylabel("Count")
    pyplot.yscale('log')
    pyplot.show()
    
    
    print("2.d")
    txt_vocab_list, trigrams_count_vocab = part_d(txt_no_punc, words_list)
  
  
  
  
    return txt_vocab_list, trigrams_count_vocab

txt_vocab_list, trigrams_count_vocab = second_question(raw_txt)

def third_question(trigrams_count_vocab):
  
  
  bigrams = set((b,c) for a,b,c in trigrams_count_vocab)
  print("Bigrams set ", len(bigrams))
  
  cnt = 1
  for x in bigrams:
    cnt += 1
    if cnt>10:
      break
    print(x)
  
  
  bigrams = np.array(list(bigrams))
  
  
  
  
  trigrams_items_np = np.array(list(trigrams_count_vocab.keys()))
  trigrams_values_np = np.array(list(trigrams_count_vocab.values()))
  
  idx = 0
  print(trigrams_items_np.shape, trigrams_values_np.shape, bigrams.shape)
  
  for idx in range(4):
    print(trigrams_items_np[idx][1:], " AA ", trigrams_values_np[idx])
  
  
  bigrams_dict = {}
  
  for trigram, count in trigrams_count_vocab.items():
    key = trigram[1:]
    if key in bigrams_dict:
      bigrams_dict[key] += count
    else:
      bigrams_dict[key] = count
    
  
  
  return bigrams_dict

cnt = 1
for x in trigrams_count_vocab:
  cnt += 1
  if cnt > 10:
    break
  
  print(x)

sum(trigrams_count_vocab.values())

bigrams_count_recomputed = third_question(trigrams_count_vocab)


print("Total bigrams : ", sum(bigrams_count_recomputed.values()))
cnt = 1
for x, y in bigrams_count_recomputed.items():
  cnt += 1
  if cnt>10:
    break
  print(x, y)

bigrams_vocab = ngrams(txt_vocab_list, 2)
  
 
#print("Num of trigrams based on vocabulary: ", sum(1 for x in trigrams_vocab))


cnt = 1
for x in bigrams_vocab:
  cnt += 1
  if cnt > 10 :
    break
  print(x)


bigrams_count = Counter(bigrams_vocab)
print("Num of bigrams based on vocabulary : ", sum(bigrams_count.values()), sum(1 for x,y in bigrams_count.items()))

def recompute_bigrams(ngrams_count):
  
  #print("Num of ngrams based on vocabulary : ", sum(ngrams_count.values()), sum(1 for x,y in ngrams_count.items()), ngrams_count.most_common(10)) 
  print("Begin recomputation..") 
 
  ngrams_dict = {}
  
  for ngram, count in ngrams_count.items():
    key = ngram[1:]
    if key in ngrams_dict:
      ngrams_dict[key] += count
    else:
      ngrams_dict[key] = count
  
  
  return ngrams_dict

unigrams_count_recomputed = recompute_bigrams(bigrams_count)
print("Num of ngrams based on recomputation : ", sum(unigrams_count_recomputed.values()), sum(1 for x,y in unigrams_count_recomputed.items()))

unigrams_vocab = ngrams(txt_vocab_list, 1)
  
 
#print("Num of trigrams based on vocabulary: ", sum(1 for x in trigrams_vocab))


cnt = 1
for x in unigrams_vocab:
  cnt += 1
  if cnt > 10 :
    break
  print(x)

  


unigrams_count = Counter(unigrams_vocab)
print(unigrams_count['adjourned'])

print("Num of unigrams based on vocabulary : ", sum(unigrams_count.values()), sum(1 for x,y in unigrams_count.items()))

print("Most common unigrams: ", unigrams_count.most_common(10))

len(txt_vocab_list)

def unigram_prob(unigrams_count, vocab_set, num_words_corpus):
  
  unigrams_count_counts = Counter(dict(unigrams_count).values())
  
  w = len(vocab_set)
  
  vocab_set = np.array(list(vocab_set))
  print(vocab_set[:5], vocab_set.shape)
  
  unigrams_np = np.array(list(unigrams_count.keys()))
  unigrams_np = np.reshape(unigrams_np, (-1))
  print(unigrams_np[:5], unigrams_np.shape)
  
  n0_list = np.setdiff1d(vocab_set, unigrams_np)
  n0 = n0_list.shape[0]
  
  assert n0 == (vocab_set.shape[0] - unigrams_np.shape[0])
  print("N_0 : ", n0)
  
  
  
  print("Number of unigrams with counts-count 1 : ", unigrams_count_counts[1])
  
  b_uni = unigrams_count_counts[1] / (unigrams_count_counts[1] + 2*unigrams_count_counts[2])
  
  print("B_uni : ", b_uni)
  
  unigrams_prob_dict = {}
  
  
  
      
  for unigram, count in unigrams_count.items():
    val = max((count - b_uni) / num_words_corpus, 0) + ((b_uni * (w - n0)) / (num_words_corpus * w))
    
    unigrams_prob_dict[unigram] = val
    
  
  temp_val = ((b_uni * (w - n0)) / (num_words_corpus * w))
  for unigram in n0_list:
    val = ((b_uni * (w - n0)) / (num_words_corpus * w))
    
    unigrams_prob_dict[unigram] = val
  
  
  
  print("Sum : ", sum(unigrams_prob_dict.values()))
  
  return unigrams_prob_dict, temp_val

def bigram_prob(bigrams_count, vocab_set, num_words_corpus, unigrams_count, unigrams_prob_dict, temp_val):
  
  bigrams_count_counts = Counter(dict(bigrams_count).values())
  
  w = len(vocab_set)
  
  vocab_set = np.array(list(vocab_set))
  #print(vocab_set[:5], vocab_set.shape)
  
  bigrams_np = np.array(list(bigrams_count.keys()))
  bigrams_np = np.reshape(bigrams_np, (-1, 2))
  
  first_word_bigram = bigrams_np[:, :1]
  first_word_bigram = np.reshape(first_word_bigram, (-1))
  print(first_word_bigram.shape)
  first_word_counter = Counter(first_word_bigram)
  
  prev_word_list = set(first_word_bigram)
  
  print("Unique ", len(prev_word_list))
  
 # print("Check ", len(set(bigrams_np)), bigrams_np.shape)
  
  second_word_bigram = bigrams_np[:, 1:]
  second_word_bigram = np.reshape(second_word_bigram, (-1))

  print(first_word_bigram.shape, first_word_bigram[:5])
  
  b_bi = bigrams_count_counts[1] / (bigrams_count_counts[1] + 2*bigrams_count_counts[2])
  
  bigrams_prob_dict = {}
  
  n0_bigrams_dict = {}
  
    

  print("Begin...")
  cnt = 0
  for bigram, count in bigrams_count.items():
    
    cnt += 1
    
    
    prev_word = bigram[:1]
    next_word = bigram[1:]
    
    #print(prev_word[0], np.where(first_word_bigram == prev_word[0]))
    #words_in_bigram = np.array(second_word_bigram[np.where(first_word_bigram == prev_word)])
    #words_in_bigram = np.unique(words_in_bigram)
    #print(words_in_bigram.shape)
    
    #print("TTT", prev_word, first_word_counter.get(prev_word[0]))
    
    n_0 = w - first_word_counter.get(prev_word[0])
    
    n_v = unigrams_count[prev_word]

    prob_uni_w = unigrams_prob_dict[next_word]
    #print("Prob: ", prob_uni_w)
    
    val = max( ((count - b_bi) / n_v), 0) + ( (b_bi * (w - n_0) * prob_uni_w) / n_v)
    
    bigrams_prob_dict[bigram] = val
    
    
    
    if cnt%100000 == 0 or prev_word[0] == "babies":
      print("Bigram no. : ", cnt, bigram, bigrams_prob_dict[bigram], first_word_counter.get(prev_word[0]), prev_word)
    
    #print(bigrams_np[idx])
    #print(words_in_bigram.shape)
  
  
  #test_val = (w - first_word_counter.get("hordes")) * (( (b_bi * first_word_counter.get("hordes")) * temp_val) / unigrams_count.get(('hordes',)))
  #print("Teststs", test_val)
  #test_val += sum([y for x,y in bigrams_prob_dict.items() if "adjourned" == x[0]])
  print("B_bi ", b_bi)
  
  print("Sum : ", sum([y for x,y in bigrams_prob_dict.items() if "babies" == x[0]]))
  #print(test_val)
  
  #assert n1 == (vocab_set.shape[0] - bigrams_np.shape[0])
  #print("N_1 : ", n1)
  
  print("Testing for one specific v........")
  
  grow_prob = {}
  grow_bigram_list = [('grow', x,) for x in vocab_set]
  print(grow_bigram_list[:5], len(grow_bigram_list))
  
  for bigram in grow_bigram_list:
    
    prev_word = bigram[:1] 
    next_word = bigram[1:]
    
    #print(prev_word[0], np.where(first_word_bigram == prev_word[0]))
    #words_in_bigram = np.array(second_word_bigram[np.where(first_word_bigram == prev_word)])
    #words_in_bigram = np.unique(words_in_bigram)
    #print(words_in_bigram.shape)
    
    #print("TTT", prev_word, first_word_counter.get(prev_word[0]))
    
    n_0 = w - first_word_counter.get(prev_word[0])
    
    n_v = unigrams_count[prev_word]

    if next_word in unigrams_prob_dict.keys():
      prob_uni_w = unigrams_prob_dict[next_word]
    else:
      prob_uni_w = 0.0
    #print("Prob: ", prob_uni_w)
    
    if bigram in bigrams_count.keys():
      val = max( ((bigrams_count.get(bigram) - b_bi) / n_v), 0) + ( (b_bi * (w - n_0) * prob_uni_w) / n_v)
    else:
      val = ( (b_bi * (w - n_0) * prob_uni_w) / n_v)
      
    grow_prob[bigram] = val
    
    
  
  print("Sum for grow: ", sum([y for x,y in grow_prob.items() if "grow" == x[0]]))
  
  return bigrams_prob_dict

vocab_set = set(line.strip() for line in open('vocabulary'))

#print(len(txt_vocab_list))
unigrams_prob_dict, temp_val = unigram_prob(unigrams_count, vocab_set, len(txt_vocab_list))

bigrams_prob_dict = bigram_prob(bigrams_count, vocab_set, len(txt_vocab_list), unigrams_count, unigrams_prob_dict, temp_val)


def preprocess_test(raw_txt, vocab_set):
  
  print("Size of vocabulary : ", len(vocab_set))
  
  txt_no_punc = raw_txt.translate(str.maketrans('', '', string.punctuation))
  
  words_list = re.findall(r'\w+', txt_no_punc)


  num_unique_words = len(set(words_list))
  words_count = Counter(words_list)

  top_10_words = words_count.most_common(10)

  print("Most common words in test corpus : ", top_10_words)

  sentences = txt_no_punc.split("\n")
  sentences_marked = np.array(['<s> ' + s + ' </s>' for s in sentences] )
  print(sentences_marked[:5], sentences_marked.shape)
  #txt_no_punc = ' '.join(sentences_marked)

  print("Test passed")


  words_list = [x.split() for x in sentences_marked]
  print("Marked text ", len(words_list), words_list[:5]) 
  
  txt_vocab_list = np.array([[x if x in vocab_set else '<unk>' for x in y] for y in words_list])
  
  print("Result : ", txt_vocab_list[:2])
  
  return txt_vocab_list

def log_likelihood(bigrams_test_vocab, sentence_sizes, num_sentences, bigrams_prob_dict):
  
  bigrams_test_vocab = np.array([np.array(list(x)) for x in bigrams_test_vocab])
  
  print("test ", bigrams_test_vocab[:3])
  
  
  print( (bigrams_test_vocab[0][2][0], bigrams_test_vocab[0][2][1] ))
  
  ll = sum( sum(np.log(bigrams_prob_dict[(bigrams_test_vocab[0][2][0], bigrams_test_vocab[0][2][1])]) if (bigrams_test_vocab[0][2][0], bigrams_test_vocab[0][2][1]) in bigrams_prob_dict.keys() else 0 for i in range(sentence_sizes[k]+1) ) for k in range(num_sentences))
    
  print("Log likelihood : ", ll)  
    
  return ll

def perplexity(raw_txt_test, bigrams_prob_dict, vocab_set):
  
 
  sentences_txt = preprocess_test(raw_txt_test, vocab_set)
  
  sentence_sizes = np.array([len(x)-2 for x in sentences_txt])
  
  print("Sentence sizes : ", sentence_sizes[:5], np.sum(sentence_sizes))
  
  print("Number of sentences : ", sentences_txt.shape)
  
  
  bigrams_test_vocab = np.array([ngrams(sentence, 2) for sentence in sentences_txt])
  
  
  """
  cnt = 1
  for _ in bigrams_test_vocab:
    print(" NEW ")
    cnt += 1
    if cnt > 1:
      break
    for x in _:
      print(x)
  """
  
  
  
  ll = log_likelihood(bigrams_test_vocab, sentence_sizes, sentences_txt.shape[0], bigrams_prob_dict)
  
  #print("Denom ", (sentences_txt.shape[0] + np.sum(sentence_sizes)))
  
  perplexity = np.exp(-ll / (sentences_txt.shape[0] + np.sum(sentence_sizes)))
  
  print("Perplexity of test corpus : ", perplexity)

with gzip.open('test.corpus.gz', 'rt') as f:
    raw_txt_test = f.read()
    
perplexity(raw_txt_test, bigrams_prob_dict, vocab_set)
