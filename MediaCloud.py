import mediacloud.api
import json
import datetime

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
    string = ""
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


    string += ' AND media_id:1'

    return string

def count_me():
    res = mc.storyCount(filtration(keywords), 'publish_date:[2014-01-01TOO:00:00Z TO 2020-06-06T00:00:00Z]')
    return res['count'] # prints the number of stories found

def stroy_grab()
    fet
if __name__ == '__main__':
    print(filtration(keywords))
    print(count_me())