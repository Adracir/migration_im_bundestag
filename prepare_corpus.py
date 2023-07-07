import os
from xml.etree import ElementTree as ET
import pandas as pd
import re
import datetime
import time
from nltk.tokenize import sent_tokenize
import gensim.utils as gu

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_MARKERS_DIR = ROOT_DIR + '/data/session_markers.csv'
EPOCHS_DIR = ROOT_DIR + '/data/epochs.csv'

# TODO: rename and document methods in a meaningful way


def prepare_text_for_embedding_training(filename):
    with open(filename, encoding='utf8') as file:
        s = file.read()
        # TODO add lemmatization at this point?
        tokenized = []
        sents = sent_tokenize(s, language="german")
        for sent in sents:
            # TODO: gu.simple_preprocess scheint auch lowercasing zu enthalten -> sinnvoll?
            #  vielleicht eher spaCy für die Vorverarbeitung benutzen oder Code kopieren und anpassen
            #  ansonsten funktioniert das aber mega gut. egalisiert auf jeden Fall auch diese Formatierungsfehler
            #  und entfernt Satzzeichen und Zahlen (das wiederum...?)
            tokenized.append(gu.simple_preprocess(sent, min_len=1, max_len=40))
        return tokenized


def preprocess_text(ep, session, epoch_beginning_date, epoch_ending_date):
    """
    extract spoken text as well as possible from xml file downloaded from https://www.bundestag.de/services/opendata
    :param ep: election period
    :param session: number of the parliamentary session
    :return: text of the specified session
    """
    # find xml file for specific session in election period
    file_path = f'{ROOT_DIR}/data/wp{ep}/{ep:02d}{session:003d}.xml'
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
    file_path = f'{ROOT_DIR}/data/wp20/20{session:003d}-data.xml'
    if os.path.exists(file_path):
        # read xml and find sitzungsverlauf tag, containing the stenographical protocol
        data = ET.parse(file_path)
        root = data.getroot()
        text_tag = root.find('sitzungsverlauf')
        text = ''.join(text_tag.itertext())
        return text


def pure_text_to_epoch_txt(epoch_id):
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
            for i in range(1, len(os.listdir(f'{ROOT_DIR}/data/wp{ep}'))+1):
                session_text = preprocess_text(ep, i, epoch_beginning_date, epoch_ending_date)
                if session_text:
                    text_to_add = " ".join([text_to_add, session_text])
                else:
                    break
    else:
        for i in range(1, len(os.listdir(f'{ROOT_DIR}/data/wp19')) + 1):
            # check for dates
            session_text = preprocess_text(19, i, epoch_beginning_date, epoch_ending_date)
            if session_text:
                text_to_add = " ".join([text_to_add, session_text])
            else:
                break
        for a in range(1, len(os.listdir(f'{ROOT_DIR}/data/wp20')) + 1):
            session_text = extract_text_from_ep_20_xml(a)
            text_to_add = " ".join([text_to_add, session_text])
    with open(txt_file_path, 'w', encoding="utf-8") as text_file:
        text_file.write(text_to_add)
    return text_to_add


print(prepare_text_for_embedding_training('data/corpus/epoch8.txt'))
