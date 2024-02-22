import os
from xml.etree import ElementTree as ET
import pandas as pd
import re
import datetime
from nltk.tokenize import sent_tokenize
import gensim.utils as gu
from HanTa import HanoverTagger as ht

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_debate_text_for_corpus_before_20th_ep(ep, session, epoch_beginning_date, epoch_ending_date):
    """
    extract transcript of spoken text in parliamentary debate from xml file downloaded from
    https://www.bundestag.de/services/opendata
    :param ep: election period
    :param session: number of the parliamentary session
    :param epoch_beginning_date: beginning date for the epoch (50s, 60s, 70s...) to filter
    :param epoch_ending_date: ending date for the epoch (50s, 60s, 70s...) to filter
    :return: text of the specified session
    """
    # find xml file for specific session in election period
    file_path = f'data/corpus/wp{ep}/{ep:02d}{session:003d}.xml'
    if os.path.exists(file_path):
        # read xml
        data = ET.parse(file_path)
        root = data.getroot()
        # check for date
        session_date_str = root.find('DATUM').text
        session_date = datetime.datetime.strptime(session_date_str, '%d.%m.%Y')
        if epoch_beginning_date <= session_date <= epoch_ending_date:
            # find TEXT tag, containing the stenographic protocol
            text_tag = root.find('TEXT')
            text = ''.join(text_tag.itertext())
            # find session beginning and ending markers and extract only text inside
            df = pd.read_csv('data/session_markers.csv', sep=';')
            beginnings_df = df[df['type'] == 'beginning']
            beginnings = beginnings_df['text'].tolist()
            endings_df = df[df['type'] == 'ending']
            endings = endings_df['text'].tolist()
            beginning_split = False
            ending_split = False
            for beginning in beginnings:
                if re.search(beginning, text):
                    split = re.split(beginning, text)
                    text = ' '.join(split[1:])
                    beginning_split = True
                    break
            for ending in endings:
                if re.search(ending, text):
                    split = re.split(ending, text)
                    text = ' '.join(split[:-1])
                    ending_split = True
                    break
            if not beginning_split:
                print(f"NO BEGINNING FOUND FOR EP {ep} SESSION {session}")
            if not ending_split:
                print(f"NO ENDING FOUND FOR EP {ep} SESSION {session}")
            return text
        else:
            return


def extract_debate_text_from_ep_20_xml(session):
    """
    extract transcript of spoken text in parliamentary debate in 20th election period from xml file downloaded from
    https://www.bundestag.de/services/opendata
    these xml files are structured following TEI standards and thus, require
    :param session: number of the parliamentary session
    :return: text of the specified session
    """
    file_path = f'data/corpus/wp20/20{session:003d}-data.xml'
    if os.path.exists(file_path):
        # read xml and find "sitzungsverlauf"-tag, containing the stenographic protocol
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
    txt_file_path = f'data/corpus/epoch{epoch_id}.txt'
    text_to_add = ''
    # read necessary info from epochs.csv
    df = pd.read_csv('data/epochs.csv')
    epoch_df = df[df['epoch_id'] == epoch_id]
    ep_start = epoch_df.ep_start.iloc[0]
    ep_end = epoch_df.ep_end.iloc[0]
    epoch_beginning_date = datetime.datetime.strptime(epoch_df.epoch_beginning_date.iloc[0], '%d.%m.%Y')
    epoch_ending_date = datetime.datetime.strptime(epoch_df.epoch_ending_date.iloc[0], '%d.%m.%Y')
    if epoch_id != 8:
        for ep in range(ep_start, ep_end + 1):
            for i in range(1, len(os.listdir(f'{ROOT_DIR}/data/corpus/wp{ep}'))+1):
                session_text = extract_debate_text_for_corpus_before_20th_ep(ep, i, epoch_beginning_date, epoch_ending_date)
                if session_text:
                    text_to_add = " ".join([text_to_add, session_text])
                else:
                    break
    else:
        for i in range(1, len(os.listdir(f'{ROOT_DIR}/data/corpus/wp19')) + 1):
            # check for dates
            session_text = extract_debate_text_for_corpus_before_20th_ep(19, i, epoch_beginning_date, epoch_ending_date)
            if session_text:
                text_to_add = " ".join([text_to_add, session_text])
            else:
                break
        for a in range(1, len(os.listdir(f'{ROOT_DIR}/data/corpus/wp20')) + 1):
            session_text = extract_debate_text_from_ep_20_xml(a)
            text_to_add = " ".join([text_to_add, session_text])
    with open(txt_file_path, 'w', encoding="utf-8") as text_file:
        text_file.write(text_to_add)
    return text_to_add


def prepare_text_for_embedding_training(filepath, lemmatize=False):
    """
    take a txt file with only the necessary text and prepares it for the generation of word embeddings by performing
    the following steps:
    - read the file
    - tokenize by sentences
    - unite words split by a newline character
    - tokenize per words
    - optionally lemmatize each word
    :param filepath: filepath of the txt file
    :param lemmatize: boolean to decide whether the words should be lemmatized or not
    :return: nested list of sentences with words/lemmas
    """
    print('...starting to prepare text...')
    with open(filepath, encoding='utf8') as file:
        s = file.read()
        tokenized = []
        sents = sent_tokenize(s, language="german")
        print(f'{len(sents)} sents extracted')
        i = 0
        hanover = ht.HanoverTagger('morphmodel_ger.pgz')
        for sent in sents:
            # first unite words split by line breaks
            sent = sent.replace('-\n', '')
            # code copied from gu.simple_preprocess, only difference: lowercasing not done,
            # as the cases might encode important semantic information in German language
            # also includes optional lemmatization using hanover lemmatizer
            tokens = [
                hanover.analyze(token)[0] if lemmatize
                else token for token in gu.tokenize(sent, lower=False, deacc=False, errors='ignore')
                if 1 <= len(token) <= 40 and not token.startswith('_')
            ]
            tokenized.append(tokens)
            i = i + 1
            print(f'\rPrepared sent No. {i}', end="")
        return tokenized
