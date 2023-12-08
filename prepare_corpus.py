import os
from xml.etree import ElementTree as ET
import pandas as pd
import re
import datetime
import time
from nltk.tokenize import sent_tokenize
import gensim.utils as gu # musste erst über pip install gensim installiert werden!
from HanTa import HanoverTagger as ht  # musste erst über pip install hanta installiert werden!
import sys

import utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_MARKERS_DIR = ROOT_DIR + '/data/session_markers.csv'
EPOCHS_DIR = ROOT_DIR + '/data/epochs.csv'

# TODO: rename and document methods in a meaningful way


# TODO: reverse/improve logic. maybe include loop in method and only filter by epoch limiting dates
def extract_debate_for_corpus_before_20th_ep(ep, session, epoch_beginning_date, epoch_ending_date):
    """
    extract transcript of spoken text in parliamentary debate from xml file downloaded from https://www.bundestag.de/services/opendata
    :param ep: election period
    :param session: number of the parliamentary session
    :param epoch_beginning_date: beginning date for the epoch (50s, 60s, 70s...) to filter
    :param epoch_ending_date: ending date for the epoch (50s, 60s, 70s...) to filter
    :return: text of the specified session
    """
    # find xml file for specific session in election period
    file_path = f'{ROOT_DIR}/data/corpus/wp{ep}/{ep:02d}{session:003d}.xml'
    if os.path.exists(file_path):
        # read xml
        data = ET.parse(file_path)
        root = data.getroot()
        # check for date
        session_date_str = root.find('DATUM').text
        session_date = datetime.datetime.strptime(session_date_str, '%d.%m.%Y')
        if epoch_beginning_date <= session_date <= epoch_ending_date:
            # find TEXT tag, containing the stenographical protocol
            text_tag = root.find('TEXT')
            text = ''.join(text_tag.itertext())
            # find session beginning and ending markers and extract only text inside of these
            df = pd.read_csv(SESSION_MARKERS_DIR, sep=';')
            beginnings_df = df[df['type'] == 'beginning']
            beginnings = beginnings_df['text'].tolist()
            endings_df = df[df['type'] == 'ending']
            endings = endings_df['text'].tolist()
            beginning_splitted = False
            ending_splitted = False
            for beginning in beginnings:
                if re.search(beginning, text):
                    splitted = re.split(beginning, text)
                    text = ' '.join(splitted[1:])
                    beginning_splitted = True
                    break
            for ending in endings:
                if re.search(ending, text):
                    splitted = re.split(ending, text)
                    text = ' '.join(splitted[:-1])
                    ending_splitted = True
                    break
            if not beginning_splitted:
                print(f"NO BEGINNING FOUND FOR EP {ep} SESSION {session}")
            if not ending_splitted:
                print(f"NO ENDING FOUND FOR EP {ep} SESSION {session}")
            return text
        else:
            return


def extract_text_from_ep_20_xml(session):
    file_path = f'{ROOT_DIR}/data/corpus/wp20/20{session:003d}-data.xml'
    if os.path.exists(file_path):
        # read xml and find sitzungsverlauf tag, containing the stenographical protocol
        data = ET.parse(file_path)
        root = data.getroot()
        text_tag = root.find('sitzungsverlauf')
        text = ''.join(text_tag.itertext())
        return text


def pure_text_to_epoch_txt(epoch_id):
    """
    create corpus base by extracting text from the different xml files and writing it to txt file for the resp. epoch
    :param epoch_id: number signifying an historical epoch defined in epochs.csv
    :return: raw text for the whole epoch
    """
    # create txt file
    txt_file_path = f'{ROOT_DIR}/data/corpus/epoch{epoch_id}.txt'
    text_to_add = ''
    # read necessary info from epochs.csv
    df = pd.read_csv(EPOCHS_DIR)
    epoch_df = df[df['epoch_id'] == epoch_id]
    ep_start = epoch_df.ep_start.iloc[0]
    ep_end = epoch_df.ep_end.iloc[0]
    epoch_beginning_date = datetime.datetime.strptime(epoch_df.epoch_beginning_date.iloc[0], '%d.%m.%Y')
    epoch_ending_date = datetime.datetime.strptime(epoch_df.epoch_ending_date.iloc[0], '%d.%m.%Y')
    if epoch_id != 8:
        for ep in range(ep_start, ep_end + 1):
            for i in range(1, len(os.listdir(f'{ROOT_DIR}/data/corpus/wp{ep}'))+1):
                session_text = extract_debate_for_corpus_before_20th_ep(ep, i, epoch_beginning_date, epoch_ending_date)
                if session_text:
                    text_to_add = " ".join([text_to_add, session_text])
                else:
                    break
    else:
        for i in range(1, len(os.listdir(f'{ROOT_DIR}/data/corpus/wp19')) + 1):
            # check for dates
            session_text = extract_debate_for_corpus_before_20th_ep(19, i, epoch_beginning_date, epoch_ending_date)
            if session_text:
                text_to_add = " ".join([text_to_add, session_text])
            else:
                break
        for a in range(1, len(os.listdir(f'{ROOT_DIR}/data/corpus/wp20')) + 1):
            session_text = extract_text_from_ep_20_xml(a)
            text_to_add = " ".join([text_to_add, session_text])
    with open(txt_file_path, 'w', encoding="utf-8") as text_file:
        text_file.write(text_to_add)
    return text_to_add


def prepare_text_for_embedding_training(filename, lemmatize=False):
    print('...starting to prepare text...')
    with open(filename, encoding='utf8') as file:
        s = file.read()
        tokenized = []
        sents = sent_tokenize(s, language="german")
        print(f'{len(sents)} sents extracted')
        i = 0
        hannover = ht.HanoverTagger('morphmodel_ger.pgz')
        for sent in sents:
            # first unite words split by line breaks
            sent = sent.replace('-\n', '')
            # code copied from gu.simple_preprocess, only difference: lowercasing not done,
            # as the cases encode important semantic information in German language
            # also includes optional lemmatization using hannover lemmatizer
            tokens = [
                hannover.analyze(token)[0] if lemmatize else token for token in gu.tokenize(sent, lower=False, deacc=False, errors='ignore')
                if 1 <= len(token) <= 40 and not token.startswith('_')
            ]
            tokenized.append(tokens)
            i = i + 1
            print(f'\rPrepared sent No. {i}', end="")
        return tokenized


def print_contexts_for_word_from_lemmatized_corpus(word, epoch):
    # find path to corpus
    corpus_path = f'data/corpus/epoch{epoch}_prepared_lemma'
    # unpickle corpus
    corpus = utils.unpickle(corpus_path)
    sents_containing_word = []
    # search lists for word
    for sent in corpus:
        if word in sent:
            sents_containing_word.append(sent)
    print(f'{len(sents_containing_word)} Sätze mit Wort {word} gefunden:')
    for s in sents_containing_word:
        print(f'{sents_containing_word.index(s)}. {s}\n')
    return sents_containing_word


# print_contexts_for_word_from_lemmatized_corpus("Auslän", 6)
'''start = time.time()
print(prepare_text_for_embedding_training('data/corpus/testepoch.txt', True))
end = time.time()
print(f'time taken: {end-start} seconds')
# 13000 words = 7.36sec'''

