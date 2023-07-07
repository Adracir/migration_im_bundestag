import pandas as pd
import time
import psycopg2
import requests
import os


def print_first_speech_from_csv_by_session(session, electoral_term):
    df = pd.read_csv('data/speeches.csv')
    df1 = df[(df['session'] == session) & (df['electoralTerm'] == electoral_term)]
    first_speech_content = df1.iloc[-1].speechContent
    print(first_speech_content)
    # time taken: 32.461670875549316 seconds


def print_first_speech_from_db_by_session(session):
    # db_connection -----------------------------------------------------------
    con_details = {
        "host": "localhost",
        "database": "next",
        "user": "postgres",
        "password": "postgres",
        "port": "5432"
    }
    con = psycopg2.connect(**con_details)

    # get data tables ---------------------------------------------------------
    df = pd.read_sql_query("select * from open_discourse.speeches", con)
    df1 = df[(df['session'] == session)]
    first_speech_content = df1.iloc[0].speech_content
    print(first_speech_content)
    # time taken: 103.2304093837738 seconds


def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


if __name__ == '__main__':
    print_first_speech_from_csv_by_session(192, 14)
    for i in range(16, 114):
        download(f"https://www.bundestag.de/resource/blob/879944/86559dfdad1e7304d92fae71974ad18d/20{i:003d}-data.xml", dest_folder="data/wp20_test")
    # url = 'https://search.dip.bundestag.de/api/v1/plenarprotokoll?f.zuordnung=BT&f.datum.start=2001-10-11&f.datum.end=2001-10-12&apikey=rgsaY4U.oZRQKUHdJhF9qguHMkwCGIoLaqEcaHjYLF'
    # response = requests.get(url)
    # print(response.status_code)
    # print(response.json())