import json
import numpy as np

train_json="data/eng_laptop/train.json"
data_dict = {}

with open(train_json, "r") as fn:
    data_dict = json.load(fn)

valence_list = []
arousal_list = []

for data in data_dict:
    valence_list.append(data["Valence"])
    arousal_list.append(data["Arousal"])


valence_list = np.array(valence_list)
arousal_list = np.array(arousal_list)

pcc = np.corrcoef(valence_list, arousal_list)
print(pcc)
