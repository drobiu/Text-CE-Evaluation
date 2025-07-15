import unicodedata
import html
import pandas as pd
import os


def unescape_characters(text):
    return text.encode().decode('unicode-escape')


def preprocess_data(row, data_split='train'):
    
    idx = data_split+"_"+str(row.name)

    # remove accents and non-latin characters
    text = unescape_characters(row.sentence)
    text = html.unescape(text)
    # text = row.sentence.replace('\u2019', "'").replace('—', '-').replace('½', '.5').replace('¾', '.75')
    text = unicodedata.normalize('NFD', text)\
                                       .encode('ascii', 'ignore')\
                                       .decode("utf-8")

    label = row.label
    
    new_row = {'id':idx, 'text':row.sentence, 'label':label}
    return new_row


def make_dataset(data_fold):
    
    # load data (train)
    train_df = pd.read_csv(os.path.join(data_fold, 'train_c.csv'), encoding='utf-8')

    # load data (test)
    test_df = pd.read_csv(os.path.join(data_fold, 'test_c.csv'), encoding='utf-8')

    # make the dataset
    train_df = train_df.apply(lambda row: preprocess_data(row, data_split='train'), axis=1)
    train_df = pd.DataFrame(train_df.tolist())

    test_df = test_df.apply(lambda row: preprocess_data(row, data_split='test'), axis=1)
    test_df = pd.DataFrame(test_df.tolist())
    
    # split train/valid
    val_frac = 0.2

    valid_df = train_df.groupby('label').sample(frac=val_frac, random_state=42).sample(frac=1, random_state=42)
    train_df = train_df[~train_df.id.isin(valid_df.id)]
    
    return train_df, valid_df, test_df