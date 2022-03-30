import json
import os
import torch

class Tag_ID_Converter:

	tag_list = []
	id_to_tag = {}
	tag_to_id = {}
	pad_id = 0

	def __init__(self, PATH_dir, data):
		
		self.id_to_tag[self.pad_id] = "[PAD]"
		self.tag_to_id["[PAD]"] = self.pad_id

		for file in data:
			path = os.path.join(PATH_dir, file)

			with open (path, 'r') as f:
				self.tag_list.extend(json.load(f))
		
		self.tag_list = list(set(self.tag_list))

		for i, tag in enumerate(self.tag_list):
			self.id_to_tag[i+1] = tag
			self.tag_to_id[tag] = i + 1

	def convert_tag_to_id_list(self, taglist):
		return [self.tag_to_id[tag] for tag in taglist]
	
	def convert_id_to_tag_list(self, idlist):
		return [self.id_to_tag[id] for id in idlist]

	def make_batch(self, labels, max_len):
		batch_labels = []
		special_token = self.pad_id
		
		for taglist in labels:
			sample_label = [special_token] + self.convert_tag_to_id_list(taglist)[:max_len - 2] + [special_token]
			sample_label += [special_token] * max(0, max_len - len(sample_label))
			batch_labels.append(sample_label)
			
		return torch.tensor(batch_labels)

		


