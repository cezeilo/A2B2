import string
import requests
import csv
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

#Topic Length: 144
#Extract all the topics from the poemhunter website. This will be used to collect poems based on their topics.
def get_poem_topics():
    topics = []

    url = 'https://www.poemhunter.com/poem-topics/'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    topics_html = soup.body.find_all("a", class_='phABClink')
    for element in topics_html:
        topic_name = element.contents[0]
        topics.append(str.lower(topic_name))
    return topics


#Extracts poems into the following dictionary format:
#{Poem Topic: [{id: Poem ID_1, poem: poem_content}, {id: Poem ID_2, poem: poem_content}, ...]
#Poem topic: A String, Poem ID: An Integer, Poem content: A string
def extract_poems(topics):
    base_url = 'https://www.poemhunter.com/poems/'

    base_topic_url = 'https://www.poemhunter.com'

    poem_data = {}
    id = 0

    written_topics = get_written_topics()

    for idx, topic in enumerate(topics):
        poem_data[topic] = []
        topic_url = base_url + topic + '/'

        #It is a bit harder to dynamically find the number of pages (we would have to switch to selenium since
        #beautiful soup has a hard time finding JS elements and the page number is wihtin a JS element)
        #So we use a hard number
        num_pages = 100

        if topic in written_topics:
            continue

        for i in tqdm(range(1, num_pages), desc= topic + '(' + str(idx) + '/' + str(len(topics)) + ')', leave=True):
            if i == 1:
                page = requests.get(topic_url)
            else:
                page = requests.get(topic_url + 'page-' + str(i) + '/')

            soup = BeautifulSoup(page.content, "html.parser")

            if len(soup.body.find_all('div', class_='phLink')) == 0:
                break

            #Need to get poem body links, click them then extract the contents, and move to next page if needed

            topic_poem_bodies = soup.body.find_all('div', class_='phLink')
            for poem_body in topic_poem_bodies:
                specific_poem_href = poem_body.find('a')['href']
                poem_url = base_topic_url + specific_poem_href
                poem_page = requests.get(poem_url)
                poem_soup = BeautifulSoup(poem_page.content, "html5lib")


                if poem_soup.find('div', {'class':'phContent phcText'}) == None:
                    print('Invalid text')
                    print(poem_url)
                    continue
                poem_html_content = poem_soup.find('div', {'class':'phContent phcText'}).p
                poem_text = poem_html_content.text

                #Clean poem text by removing indents, tabs, extra spaces, etc.
                clean_poem = ''
                for line in poem_html_content.encode_contents().decode('utf-8').split('<br/>'):
                    clean_poem += line.strip() + '\n'

                #Delete the two extra newline charachter
                clean_poem = clean_poem[:-1]
                clean_poem_with_newline = repr(clean_poem)
                poem_data[topic].append({"id":id, "poem":clean_poem})
                id += 1

        #Save results dynamically
        if idx == 0:
            write_poem_topic("")
            write_poems(poem_data, True)
        else:
            write_poems(poem_data, False)
        write_poem_topic(topic)


    return poem_data

#Extract poem statistics of interest
def poem_stats(poem_data):
    i = 0
    #1. Number of poems for each topic
    for topic in poem_data:
        print(topic, ': ', len(poem_data[topic]))
        i += len(poem_data[topic])

    #2. Total number of poems
    print('Total number of poems: ', i)

#Write poems to csv files for later use
#Format: "Topic, ID, Poem"
def write_poems(poem_data, header):
    filename = 'data/topic_poems.csv'
    header = ['Topic', 'ID', 'Poem']
    already_written = get_written_topics()

    with open(filename, 'a', encoding='utf-8', newline='') as f:
        csvwriter = csv.writer(f)

        if header:
            csvwriter.writerow(header)

        for poem_topic in poem_data:
            if poem_topic in already_written:
                continue
            for poem in poem_data[poem_topic]:
                csvwriter.writerow([poem_topic, poem['id'], poem['poem']])


def write_poem_topic(topic):
    with open('data/completed_topics.txt', 'a') as f:
        f.write(topic + '\n')


def get_written_topics():
    topics = []
    with open('data/completed_topics.txt', 'r') as f:
        topics.append(f.readline().strip())
    return topics

if __name__ ==  '__main__':
    poem_topics = get_poem_topics()
    poems = extract_poems(poem_topics)

    poem_stats(poems)
    #write_poems(poems)
