import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
vectorizer = AutoModel.from_pretrained('./vectorizer')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


def replace_values_l(value):
    if ' л' in value:
        value = value.replace(' л', '000 мл')
        value = value.replace('.0', '')
        return value
    elif 'л' in value:
        pattern = r'(\d+(?:\.\d+)?)\s*л\b'
        matches = re.findall(pattern, value, flags=re.IGNORECASE)
        for match in matches:
            replacement = f"{float(match) * 1000:.0f} мл"
            value = re.sub(
                fr'({match})\s*л\b',
                replacement,
                value,
                flags=re.IGNORECASE
            )
        return value
    else:
        return value


def replace_values_kg(value):
    if ' кг' in value:
        value = value.replace(' кг', '000 г')
        value = value.replace('.0', '')
        return value
    elif 'кг' in value:
        pattern = r'(\d+(?:\.\d+)?)\s*кг\b'
        matches = re.findall(pattern, value, flags=re.IGNORECASE)
        for match in matches:
            replacement = f"{float(match) * 1000:.0f} г"
            value = re.sub(
                fr'({match})\s*кг\b',
                replacement,
                value,
                flags=re.IGNORECASE
            )
        return value
    else:
        return value


def string_filter_emb(string):
    string = string.lower()
    string = replace_values_kg(replace_values_l(string))
    string = re.sub(r'[^a-zo0-9а-я\s:]', ' ', string)
    string = re.sub(r'(?<=[а-я])(?=[a-z])|(?<=[a-z])(?=[а-я])', ' ', string)
    string = re.sub(r'(?<=[а-яa-z])(?=\d)|(?<=\d)(?=[а-яa-z])', ' ', string)
    return string


class InfloatVectorizer():
    def __init__(self,
                 tokenizer=tokenizer,
                 vectorizer=vectorizer):

        self.tokenizer = tokenizer
        self.model = vectorizer

    def fit(self, X=None):
        pass

    def transform(self, corpus):
        batch_dict = self.tokenizer(
            corpus,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        last_hidden = outputs.last_hidden_state.masked_fill(
            ~batch_dict['attention_mask'][..., None].bool(), 0.0
        )
        embeddings = (last_hidden.sum(dim=1)
                      / batch_dict['attention_mask'].sum(dim=1)[..., None])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy()


class DistanceRecommender():
    def __init__(self,
                 vectorizer,
                 simularity_func,
                 text_prep_func):
        self.vectorizer = vectorizer
        self.simularity_counter = simularity_func
        self.preprocessing = text_prep_func

    def fit(self,
            product_corpus,
            name_column,
            id_column,
            save_to_dir=False):
        preprocessed_corpus = (
            product_corpus[name_column].apply(
                self.preprocessing
            ).values.tolist()
        )
        self.vectorizer.fit(preprocessed_corpus)
        self.product_matrix = self.vectorizer.transform(preprocessed_corpus)
        self.product_index_to_id = {i: product_corpus.loc[i, id_column] for i in range(len(product_corpus))}
        if save_to_dir:
            np.save('product_matrix.npy', self.product_matrix)

            with open('product_index_to_id.json', 'w') as file:
                json.dump(self.product_index_to_id, file, cls=NumpyEncoder)

    def from_pretrained(
        self,
        product_matrix_path='./model_files/product_matrix.npy',
        product_index_to_id_dict_path='./model_files/product_index_to_id.json'
    ):
        self.product_matrix = np.load(product_matrix_path)

        with open(product_index_to_id_dict_path, 'rb') as file:
            self.product_index_to_id = json.load(file)

    def recommend(self,
                  dealer_corpus: list[dict]):
        preprocessed_corpus = dealer_corpus.apply(
            self.preprocessing
        ).values.tolist()
        vectors = self.vectorizer.transform(preprocessed_corpus)
        sims = self.simularity_counter(vectors, self.product_matrix)

        result = []
        for vec in sims:
            result += [[self.product_index_to_id[str(index)] for index in vec.argsort()[::-1]]]
        return np.array(result)


def dealerprice_table(table_path='marketing_dealerprice.csv',
                      product_id_column='product_key',
                      dealer_id_column='dealer_id',
                      read_params={'on_bad_lines': "skip",
                                   'encoding': 'utf-8',
                                   'sep': ';'}):
    '''
    Функция принимает:
    .Путь к csv файлу, содержащему результаты парсинга.
    .Названия колонок с id товаров и id дилеров
    .Параметры чтения csv можно указать, если вдруг они изменятся.
    '''

    table_csv = pd.read_csv(table_path, **read_params)
    table_csv = table_csv.sort_values(
        'date', ascending=False
    ).drop_duplicates(
        subset=[
            product_id_column,
            dealer_id_column
        ]
    )
    return table_csv


model = DistanceRecommender(
    vectorizer=InfloatVectorizer(
        tokenizer=tokenizer,
        vectorizer=vectorizer
    ),
    simularity_func=cosine_similarity,
    text_prep_func=string_filter_emb
)

model.from_pretrained()


names = ['Герметик акриловый  цвет белый , ф/п 600 мл. (12 штук )',
         'Гель эконом-класса для мытья  посуды вручную. С ароматом яблокаCooky Apple Eконцентрированное средство / 5 л ПЭТ',
         'Средство для удаления ржавчины и минеральных отложений щадящего действияBath Acid  концентрат 1:200-1:500 / 0,75 л ',
         'Антисептик многофункциональный ФБС, ГОСТ / 5 л',
         'Гелеобразное средство усиленного действия для удаления ржавчины и минеральных отложенийBath Extraконцентрат 1:10-1:100 / 0,75 л']

names = pd.Series(names)

print(model.recommend(names))
