import torch
from tqdm import tqdm
from transformers import AutoTokenizer 
from transformers import BertModel 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import pandas as pd
import numpy as np
import os

from connect_database import get_data_with_sqlalchemy
from preprocessing_data import del_symbols, del_stopwords, lemmatize 

posts_df = get_data_with_sqlalchemy('posts', 200000)

posts_df['text'] = post_df['text'].apply(del_symbols)\
                                  .apply(del_stopwords)\
                                  .apply(lemmatize)

# скачиваем предобученную BERT для получения ембедингов
tokenizer, model = AutoTokenizer.from_pretrained('bert-base-cased'), BertModel.from_pretrained('bert-base-cased')

# создаем датасет для постов
class PostDataset(Dataset):
    def __init__(self, texts, tokenizer):
        super().__init__()

        self.texts = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return {'input_ids': self.texts['input_ids'][idx], 'attention_mask': self.texts['attention_mask'][idx]}

    def __len__(self):
        return len(self.texts['input_ids'])
    
    
dataset = PostDataset(posts_df['text'].values.tolist(), tokenizer)# токенизируем слова

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # для приведения текстов к одинаковой длинне

loader = DataLoader(dataset, batch_size=32, collate_fn=data_collator, pin_memory=True, shuffle=False)


@torch.inference_mode()# отключаем расчет градиента
def get_embeddings_labels(model, loader):
    '''Получаем ембединги с помощью модели'''

    model.eval()
    
    total_embeddings = []
    
    for batch in tqdm(loader):
        batch = {key: batch[key].to(device) for key in ['attention_mask', 'input_ids']}

        embeddings = model(**batch)['last_hidden_state'][:, 0, :]

        total_embeddings.append(embeddings.cpu())

    return torch.cat(total_embeddings, dim=0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

embeddings = get_embeddings_labels(model, loader).numpy()

# Сохраняем ембединги
today = str(datetime.date.today())

os.chdir('BERT_embeddings')

emb_file = open(f'embeddings_{today}', 'wb')

np.save(emb_file, embeddings)

os.chdir('..')
