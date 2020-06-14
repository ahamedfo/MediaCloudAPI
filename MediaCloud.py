import mediacloud.api
import json
import datetime
from newspaper import Article
from newspaper import Config
import csv

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
    with open('ferguson-michael-brown-all-story-urls-20200613215441.csv') as csv_file:
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
    while i < 100:
        url = list[i][0]
        article = Article(url, language='en')
        article.download()
        try:
            article.parse()
        except:
            i += 1
            continue
        print(article.text)
        f = open(list[i][1], 'w')
        f.write(article.text)
        f.close()
        i += 1
if __name__ == '__main__':
    article_reader(text_reader())

