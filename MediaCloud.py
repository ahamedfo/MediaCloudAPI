import mediacloud.api
import json
import datetime
from newspaper import Article, ArticleException
from newspaper import Config
import csv
import spacy

import pandas as pd
import itertools as it
import time
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
import gensim
import pickle
import IPython
import jsonpickle

import pyLDAvis
import pyLDAvis.gensim
import warnings


nlp = spacy.load('en')

mc = mediacloud.api.MediaCloud('ebfb0ac259e3df67c108c6600420a714870f074e1781b270fe4cbd8754d4bbb4')



keywords = ['ferguson', 'michael brown', 'mike brown', 'blacklivesmatter',
            'black lives', 'blm', 'alllivesmatter', 'whitelivesmatter', 'bluelivesmatter', 'eric garner',
            'ericgarner', 'freddie gray', 'freddiegray',
            'walter scott', 'walterscott', 'tamir rice', 'tamirrice',
            'black lives matter', 'john crawford', 'johncrawford', 'tony robinson',
            'tonyrobinson', 'eric harris', 'ericharris', 'ezell ford', 'ezellford',
            'akai gurley', 'akaigurley', 'kajieme powell', 'kajiemepowell', 'tanisha anderson',
            'tanishaanderson', 'victor white', 'victorwhite', 'jordan baker', 'jordanbaker', 'jerame reid',
            'jeramereid', 'yvette smith', 'yvettesmith', 'phillip white', 'philipwhite', 'dante parker',
            'danteparker', 'mckenzie cochran', 'mckenziecochran', 'tyree woodson', 'tyreewoodson']


def filtration(keywords):
    string = "("
    for value in range(len(keywords)):
        space = False
        for i in keywords[value]:
            if i == ' ':
                space = True
        if value == len(keywords) - 1:
            if space == False:
                string += keywords[value]
            else:
                string += '\"' + keywords[value] + '\"'

        elif space == False:
            string += keywords[value] + ' OR '
        else:
            string += '\"' + keywords[value] + '\"' + ' OR '

    string += ')'

    return string


def count_me():
    res = mc.storyCount(filtration(keywords), 'publish_date:[2014-01-01T00:00:00Z TO 2020-06-06T00:00:00Z]')
    return res['count'] # prints the number of stories found

def story_grab():
    fetch_size = 1
    stories = []
    last_processed_stories_id = 0
    while len(stories) < 1:
        fetched_stories = mc.storyList(filtration(keywords),
                                       solr_filter=mc.dates_as_query_clause(datetime.date(2014, 1, 1),
                                                                            datetime.date(2020, 6, 6)),
                                       last_processed_stories_id=last_processed_stories_id, rows=fetch_size)
        stories.extend(fetched_stories)
        last_processed_stories_id = stories[-1]['processed_stories_id']
    print(json.dumps(stories))
    return json.dumps(stories)

def story_runner(storiez):

    for values in storiez:
        print(values)

def text_reader():
    with open('black-lives-matter-all-story-urls-20200621214756.csv') as csv_file:
        list_rows = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                list_rows += {(row[3], row[2])}
                line_count += 1
        #print(f'Processed {line_count} lines.')
        return list_rows

def article_reader(list):
    i = 0
    while i < 10:
        url = list[i][0]
        article = Article(url, language='en')
        article.download()
        try:
            article.parse()
        except (ArticleException, AttributeError, UnicodeError) as e:
            i += 1
            continue
        if i % 100 == 0:
            print(article.text)
        f = open(list[i][1], 'w')
        f.write(article.text)
        f.close()
        i += 1
        time.sleep(5)

def NLP_parsed_review(url_list):
    i = 0
    while i < 10:
        url = url_list[i][0]
        article = Article(url, language='en')
        article.download()
        try:
            article.parse()
            parsed_review = nlp(article.text)
            print("New Article '\n'")
            for num,sentence in enumerate(parsed_review.sents):
                print('Sentence {}:'.format(num + 1))
                print(sentence)
                print('')
        except (ArticleException, AttributeError, UnicodeError) as e:
            i += 1
            continue
        f = open(url_list[i][1], 'w')
        f.write(article.text)
        f.close()
        i += 1
        time.sleep(5)

#/data/blm/article_texts/text'
intermediate_directory = 'data/'
def text_open():
    i = 0
    string_list = []
    strings = ""
    while i < 3:
        with open('data/text' + str(i), encoding='utf_8') as f:
            for word in f:
                strings += word
            string_list.append(strings)
            #print(string_list)
            #print('\n')
        i += 1
    return string_list

def punct_space(token):
    return token.is_punct or token.is_space

def line_review(filename):
        for review in filename:
            # print(review)
            yield review.replace('\\n', '\n')

def lemmatized_sentence_corpus(filename):
    for parsed_review in nlp.pipe(line_review(filename), batch_size=1, n_threads=4):
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent if not punct_space(token)])

unigram_sentences_filepath = intermediate_directory + 'unigram_sentences_all.txt'

def runners_Sen(unigram_sentences_filepaths):

    with open(unigram_sentences_filepaths, 'w', encoding='utf_8') as f:
        for sentence in lemmatized_sentence_corpus(text_open()):
            #print(sentence)
            f.write(sentence + '\n')

    unigram_sentences = LineSentence(unigram_sentences_filepaths)

    # for unigram_sentence in unigram_sentences:
        # print(u' '.join(unigram_sentence))
        # print(u' ')

    return unigram_sentences

bigram_model_filepath = intermediate_directory + 'bigram_model_all'

def phrases():
    unigram_sentences = runners_Sen(unigram_sentences_filepath)
    bigram_model = Phrases(unigram_sentences)
    bigram_model.save(bigram_model_filepath)
    bigram_model = Phrases.load(bigram_model_filepath)
    bigram_sentences_filepath = intermediate_directory + 'bigram_model_all.txt'
    with open(bigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for unigram_sentence in unigram_sentences:
            bigram_sentence = u' '.join(bigram_model[unigram_sentence])
            f.write(bigram_sentence + '\n')

    bigram_sentences = LineSentence(bigram_sentences_filepath)
    # print(u' '.join(bigram_sentence))
    # print(u'')

    trigram_model_filepath = intermediate_directory + 'trigram_sentences_all'

    trigram_model = Phrases(bigram_sentences)
    trigram_model.save(trigram_model_filepath)
    trigram_model = Phrases.load(trigram_model_filepath)
    trigram_sentences_filepath = intermediate_directory + 'trigram_sentences_all.txt'

    with open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for bigram_sentence in bigram_sentences:
            trigram_sentence = u' '.join(trigram_model[bigram_sentence])
            f.write(trigram_sentence + '\n')

    trigram_sentences = LineSentence(trigram_sentences_filepath)

    # for trigram_sentence in trigram_sentences:
        # print(u' '.join(trigram_sentence))
        # print(u'')

    ### STOP WORDS REMOVAL ###

    trigram_reviews_filepath = intermediate_directory + 'trigram_transformed_reviews_all.txt'
    with open(trigram_reviews_filepath, 'w', encoding='utf_8') as f:
        for parsed_review in nlp.pipe(line_review(text_open()),
                                      batch_size=10000, n_threads=4):
            # lemmatize the text, removing punctuation and whitespace
            unigram_review = [token.lemma_ for token in parsed_review
                              if not punct_space(token)]

            # apply the first-order and second-order phrase models
            bigram_review = bigram_model[unigram_review]
            trigram_review = trigram_model[bigram_review]

            # remove any remaining stopwords
            all_stopwords_gensim = STOPWORDS
            #print(STOPWORDS)
            #print(trigram_review)

            trigram_review = [term for term in trigram_review if term not in all_stopwords_gensim]

            # write the transformed review as a line in the new file
            trigram_review = u' '.join(trigram_review)
            f.write(trigram_review + '\n')

    ######BAG OF WORDS CREATION #########
    trigram_dictionary_filepath = intermediate_directory + 'trigram_dict_all.dict'
    trigram_reviews = LineSentence(trigram_reviews_filepath)

    # learn the dictionary by iterating over all of the reviews
    trigram_dictionary = Dictionary(trigram_reviews)

    # filter tokens that are very rare or too common from
    # the dictionary (filter_extremes) and reassign integer ids (compactify)
    #########trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)##########
    trigram_dictionary.compactify()
    #print(trigram_dictionary)

    trigram_dictionary.save(trigram_dictionary_filepath)

    trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)

    #print(trigram_dictionary)
    return trigram_dictionary

trigram_bow_filepath = intermediate_directory + 'trigram_bow_corpus_all.mm'
trigram_dictionary = phrases()

def trigram_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """


    for review in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(review)

trigram_reviews_filepath = intermediate_directory + 'trigram_transformed_reviews_all.txt'

def B_O_wCreator(Trigrams_filepath):

    MmCorpus.serialize(trigram_bow_filepath,
                       trigram_bow_generator(Trigrams_filepath))

    trigram_bow_corpus = MmCorpus(trigram_bow_filepath)
    return trigram_bow_corpus

def trigramz():

    trigram_bow_corpus = B_O_wCreator(trigram_reviews_filepath)

    lda_model_filepath = intermediate_directory + 'lda_model_all'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # workers => sets the parallelism, and should be
        # set to your number of physical cores minus one
        lda = LdaMulticore(trigram_bow_corpus,
                           num_topics=50,
                           id2word=trigram_dictionary,
                           workers=3)

    lda.save(lda_model_filepath)

    lda = LdaMulticore.load(lda_model_filepath)
    #print(lda)
    return lda


def explore_topic(topic_number, topn=25):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
    lda = trigramz()
    print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')

    i = 0
    topic_names = {}
    for term, frequency in lda.show_topic(topic_number, topn=25):
        print( u'{:20} {:.3f}'.format(term, round(frequency, 3)))
        topic_names[i] = term
        i += 1

   # print(topic_names)
    topic_names_filepath = intermediate_directory + 'topic_names.pkl'


    with open(topic_names_filepath, 'wb') as f:
        pickle.dump(topic_names, f)

LDAvis_data_filepath = intermediate_directory + 'ldavis_prepared'

def LDA_Diagram():
    trigram_bow_corpus = MmCorpus(trigram_bow_filepath)

    lda = trigramz()

    LDAvis_prepared = pyLDAvis.gensim.prepare(lda, trigram_bow_corpus,
                                              trigram_dictionary)

    #print(LDAvis_prepared)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        f.close()


    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
        f.close()

    print(pyLDAvis.display(LDAvis_prepared).data)

Docs = 'black-lives-matter-all-story-urls-20200621214756.csv'

def text_reader(DOC):
    with open(DOC) as csv_file:
        list_rows = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                list_rows += {(row[1], row[2])}
                line_count += 1
        #print(f'Processed {line_count} lines.')
        return list_rows


if __name__ == '__main__':
    print(trigramz().get_document_topics(B_O_wCreator(trigram_reviews_filepath)))
    print(B_O_wCreator(trigram_reviews_filepath))