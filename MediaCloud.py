import mediacloud.api
import json
import datetime
from newspaper import Article, ArticleException
from newspaper import Config
import csv
import spacy

import pandas as pd
import time
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.models.ldamulticore import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
from spacy.lang.en.stop_words import STOP_WORDS
import pickle

import pyLDAvis
import pyLDAvis.gensim
import warnings
import random

nlp = spacy.load('en')

mc = mediacloud.api.MediaCloud('ebfb0ac259e3df67c108c6600420a714870f074e1781b270fe4cbd8754d4bbb4')




#--------------------------------------------------- TASK #2 -----------  YEAR OF ARTICLE AND VISUALIZATION ----------------------------------------#

docs = '/Users/ahamedfofana/PycharmProjects/MediaCloudAPI/black-lives-matter-all-story-urls-20200621214756.csv'


from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
import numpy
from gensim.matutils import hellinger


# def headline_counter(headlines, article_num=20):
#     """Headline counter creates the dictionary mapping each month to a number representing the articles in said month."""
#     dicts = {}
#     line_count = 0
#     with open(headlines) as csv_file:
#         next(csv_file)
#         csv_reader = list(csv.reader(csv_file, delimiter=','))
#         for row in csv_reader:
#             date = str(row[1])
#             if date == '':
#                 continue
#             #print(date)
#             if (date[:7]) not in dicts:
#                 dicts[row[1][:7]] = 1
#             else:
#                 dicts[row[1][:7]] += 1
#             if line_count == article_num:
#                 return dicts
#             line_count += 1
#     return dicts

# def headline_counter(headlines, article_num=1000):
#     dicts = {}
#     line_count = 0
#     with open(headlines) as csv_file:
#         next(csv_file)
#         csv_reader = list(csv.reader(csv_file, delimiter=','))
#         for row in csv_reader:
#             date = str(row[1])
#             if date == '':
#                 continue
#             #print(date)
#             if (date[:7]) not in dicts:
#                 dicts[row[1][:7]] = 1
#             else:
#                 dicts[row[1][:7]] += 1
#             line_count += 1
#         if line_count == article_num:
#             return dicts
#
#     return dicts

TOTAL_ARTICLES = 0
def increment():
    global TOTAL_ARTICLES
    TOTAL_ARTICLES = TOTAL_ARTICLES+1

dictionary_dates = {}
print(dictionary_dates)

def article_read_urls(lists,directory):
    """Reads article urls and writes text. If an article cannot be parsed we decrement the count of that month by 1 because the article will not exist in our corpus.
      corpus is sorted in chronological order during text_reader_ye() to help with our time slice"""
    i = 0
    j = 0
    while i < len(lists):
        url = lists[i][1]
        year = lists[i][0]
        article = Article(url, language='en')
        article.download()
        try:
            article.parse()
        except (ArticleException, AttributeError, UnicodeError) as e:
            i += 1
            continue
        if year == '':
            continue
        # print(date)
        if (year[:7]) not in dictionary_dates:
            dictionary_dates[year[:7]] = 1
        else:
            dictionary_dates[year[:7]] += 1
        f = open(directory + 'text' + str(j), 'w')
        f.write(article.text)
        increment()
        f.close()
        i += 1
        j += 1
        time.sleep(5)


def text_reader_ye(DOC, article_num=25):
    """text_reader_ye is sorted in chronolgical order so that we have some structure to use our time slice and we can tell where data is in the corpus. for example:
        ['2014-01'] -> 469, ['2014-02'] -> 355 -- in order to get from 2014-01 to 2014-02 we  have to run over 469 articles in the corpus"""
    with open(DOC) as csv_file:
        list_rows = []
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        line_count = 0
        for row in csv_reader:
            if str(row[1]) == '':
                continue
            if line_count == 0:
                line_count += 1
            else:
                list_rows += {(row[1], row[3])}
                line_count += 1
            if line_count == article_num:

                list_rows.sort(key=lambda r: (int(r[0][:4]), int(r[0][5:7]), int(r[0][8:11])))
                #print(list_rows)
                return list_rows




def time_slice_start_end(date_1):
    """ Grabs the time slice from the start date of our document - '2014-01' - to the requested start date.
        our time=0 will be all the months prior to the first month. time=1 is the first month"""
    time_slice = []
    start_date = date_1

    cur_date = '2014-01'
    timeframe = False
    sum = 0
    while cur_date != start_date:
        if cur_date in dictionary_dates:
            sum += dictionary_dates[cur_date]

        if int(cur_date[5:7]) + 1 <= 12:
            if int(cur_date[5:7]) + 1 < 10:
                cur_date = cur_date[:4] + '-0' + str(int(cur_date[5:7]) + 1)
            else:
                cur_date = cur_date[:4] + '-' + str(int(cur_date[5:7]) + 1)

        # otherwise increment the entire year and find the next val in the dictionary from the start of the year
        else:
            cur_date = str(int(cur_date[:4]) + 1) + '-01'
    return sum



def time_slice(date_1, date_2):
    """time_slice returns a list of integers representing the time_slice from the start date to the end date. example:
    start date = '2014-01', end_date = '2014-03'
    ['2014-01'] -> 469 articles, ['2014-02'] -> 355 articles, ['2014-03'] -> 201 articles
    time_slice = [469, 355, 201]
    Now that we have these numbers we know how many articles we should check until we are at the end of a given month because our corpus is sorted in chronological order"""

    time_slice = []
    end_date = date_2
    cur_date = date_1
    first_dayTo_Start = time_slice_start_end(date_1)
    time_slice.append(first_dayTo_Start)
    if date_1 not in dictionary_dates or date_2 not in dictionary_dates:
        return 'not in dictionary'
    while cur_date != end_date:
        if cur_date in dictionary_dates:
            time_slice.append(dictionary_dates[cur_date])

        # if month is less than 12, increment month
        if int(cur_date[5:7]) + 1 <= 12:
            if int(cur_date[5:7]) + 1 < 10:
                cur_date = cur_date[:4] + '-0' + str(int(cur_date[5:7]) + 1)
            else:
                cur_date = cur_date[:4] + '-' + str(int(cur_date[5:7]) + 1)

        #otherwise increment the entire year and find the next val in the dictionary from the start of the year
        else:
            cur_date = str(int(cur_date[:4]) + 1) + '-01'
    if end_date in dictionary_dates:
        time_slice.append(dictionary_dates[end_date])
    return time_slice


intermediate_directory = 'data/'


def punct_space(token):
    return token.is_punct or token.is_space


def line_review(article_dir):
    print('articles = ' + str())

    for article_num in range(TOTAL_ARTICLES):
        print('line_review')
        if article_num % 100 == 0:
                print("/n/n/n/n/n/n/n/n/n" + "complete" + str(article_num))
        cur_article = article_dir + "/text" + str(article_num)
        with open(cur_article, encoding="utf-8") as f:
            next(f)
            for line in f:
                yield line.replace('\\n', '\n')



unigram_sentences_filepath = intermediate_directory + 'unigram_sentences_all.txt'


def sentence_generator(article_dir):
    """
    Generator function that yields each sentence in all text files.
    """
    #print('articles = ' + str(articles_to_parse))
    for article_num in range(TOTAL_ARTICLES):
        print('sentence_generator')
        cur_article = article_dir + "/text" + str(article_num)
        with open(cur_article, encoding="utf-8") as f:
            next(f)  # skip first line.
            if article_num % 100 == 0:
                print("/n/n/n/n/n/n/n/n/n" + "complete" + str(article_num))
            data = f.read()
            corpus = nlp(data)
            print('sents_gen')
            for sent in corpus.sents:
                # filter out punctuation and whitespace from sentences.
                cur_sentence = " ".join([token.lemma_ for token in sent
                                         if not punct_space(token)])
                # TODO - Deal with the -PRON-
                yield cur_sentence

def write_all_article_sentences():
    """
    writes all sentences into one file.
    So it can be used by spaCy's LineSentence function.
    Then returns a LineSentence iterator of sentence unigrams.
    """
    print('write_all_article_sentences')
    with open(unigram_sentences_filepath, 'w', encoding="utf-8") as f:
        for sentence in sentence_generator("data"):
            print('write_all_article_sentences')
            f.write(sentence + '\n')


    # for unigram_sentence in unigram_sentences:
        # print(u' '.join(unigram_sentence))
        # print(u' ')



bigram_model_filepath = intermediate_directory + 'bigram_model_all'
trigram_dictionary_filepath = intermediate_directory + 'trigram_dict_all.dict'
trigram_dictionary_filepath = intermediate_directory + 'trigram_dict_all.dict'

from gensim.corpora import Dictionary, MmCorpus

def phrases():
    unigram_sentences = LineSentence(unigram_sentences_filepath)
    bigram_model = Phrases(unigram_sentences)
    bigram_model.save(bigram_model_filepath)
    bigram_model = Phrases.load(bigram_model_filepath)
    bigram_sentences_filepath = intermediate_directory + 'bigram_model_all.txt'
    with open(bigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for unigram_sentence in unigram_sentences:
            bigram_sentence = u' '.join(bigram_model[unigram_sentence])
            f.write(bigram_sentence)

    bigram_sentences = LineSentence(bigram_sentences_filepath)

    trigram_model_filepath = intermediate_directory + 'trigram_sentences_all'

    trigram_model = Phrases(bigram_sentences)
    trigram_model.save(trigram_model_filepath)
    trigram_model = Phrases.load(trigram_model_filepath)
    trigram_sentences_filepath = intermediate_directory + 'trigram_sentences_all.txt'

    with open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for bigram_sentence in bigram_sentences:
            trigram_sentence = ' '.join(trigram_model[bigram_sentence])
            f.write(trigram_sentence + '\n')

    trigram_sentences = LineSentence(trigram_sentences_filepath)

    ### STOP WORDS REMOVAL ###

    trigram_reviews_filepath = intermediate_directory + 'trigram_transformed_reviews_all.txt'
    with open(trigram_reviews_filepath, 'w', encoding='utf_8') as f:
        for parsed_review in nlp.pipe(line_review('data/'),
                                      batch_size=10000, n_threads=4):

            # lemmatize the text, removing punctuation and whitespace
            unigram_review = [token.lemma_ for token in parsed_review
                              if not punct_space(token)]

            # apply the first-order and second-order phrase models
            bigram_review = bigram_model[unigram_review]
            trigram_review = trigram_model[bigram_review]


            trigram_review = [term for term in trigram_review if
                              term not in STOP_WORDS and term != '-PRON-' and term != '‘' and term != '’' and term != "'s" and term != "’s"]

            # write the transformed review as a line in the new file
            trigram_review = ' '.join(trigram_review)

            ## MOVED OUTSIDE OF THE LOOP SO WE COULD GET A SINGULAR CORPUS WITH ALL THE TEXT
            # print(trigram_review)
            f.write(trigram_review + '\n')

    ######BAG OF WORDS CREATION #########
    trigram_reviews = LineSentence(trigram_reviews_filepath)

    # learn the dictionary by iterating over all of the reviews
    trigram_dictionary = Dictionary()
    trigram_dictionary.add_documents(trigram_reviews)

    # add keep_n=10000
    trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)
    trigram_dictionary.compactify()

    trigram_dictionary.save_as_text(trigram_dictionary_filepath)



trigram_bow_filepath = intermediate_directory + 'trigram_bow_corpus_all.mm'


def trigram_bow_generator(filepath, trigram_dict):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """

    for review in LineSentence(filepath):
        yield trigram_dict.doc2bow(review)


trigram_reviews_filepath = intermediate_directory + 'trigram_transformed_reviews_all.txt'


def B_O_wCreator(Trigrams_filepath, trigram_dict):
    MmCorpus.serialize(trigram_bow_filepath,
                       trigram_bow_generator(Trigrams_filepath, trigram_dict))

    trigram_bow_corpus = MmCorpus(trigram_bow_filepath)
    return trigram_bow_corpus


lda_model_filepath = intermediate_directory + 'lda_model_all'
lda_model_filepaths = intermediate_directory + 'lda_model_alls'

def trigramz(date_1, date_2):
    """We then grab the time slice and then we can run the ldaseq with the given time_slice"""

    # grab the time_slice of the documents that we need
    time_sliced = time_slice(date_1, date_2)
    print(time_sliced)
    if time_sliced == 'not in dictionary':
        raise ValueError('dates must be contained in text timeframe')
    trigram_dictionary = phrases()
    print(trigram_dictionary)
    trigram_bow_corpus = B_O_wCreator(trigram_reviews_filepath, trigram_dictionary)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # lda = LdaMulticore(trigram_bow_corpus,
        #                    num_topics=50,
        #                    id2word=trigram_dictionary,
        #                    workers=3)

        ldaseq = ldaseqmodel.LdaSeqModel(corpus=trigram_bow_corpus, id2word=trigram_dictionary, time_slice=time_sliced, num_topics=50)

    ldaseq.save(lda_model_filepaths)
    print(ldaseq)





if __name__ == '__main__':

    list_dates = text_reader_ye(docs)
    article_read_urls(list_dates, 'data/')
    write_all_article_sentences()
    #phrases()
    trigramz('2014-12', '2015-02')
    ldaseq = ldaseqmodel.LdaSeqModel.load(lda_model_filepaths)
    print(ldaseq.print_topics(time=1))
    # print(a)
    # print(len(a))





    #print(dictionary_dates)
    #print(pd.DataFrame(8,['label', 'Black Girl', 'military'], [ 'Top Five Words']))

    #article_reader(text_reader())