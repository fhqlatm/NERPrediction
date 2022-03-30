import torch
import torch.nn.functional as F
import json
from transformers import BertTokenizer
from tag_id_converter import Tag_ID_Converter

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Dataset 상속
class Dataset_NER(Dataset): 
    def __init__(self, data):
        self.x = load_data(data)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx]

def load_data(data):
    with open(data, 'r') as f:
        return json.load(f)

def ner_collate_fn(tokenizer: BertTokenizer, tag_converter: Tag_ID_Converter, batch):

    input_sentences = [sample[0] for sample in batch]
    input_labels = [sample[2] for sample in batch]

    batch_inputs = tokenizer(input_sentences, padding = True, return_tensors = "pt")
    batch_labels = tag_converter.make_batch(input_labels, max_len = batch_inputs['input_ids'].size(1))
    
    return batch_inputs, batch_labels
