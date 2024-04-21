from bpemb import BPEmb
import pandas as pd
import numpy as np

from tqdm import tqdm
bpemb_vi = BPEmb(lang="vi", vs=25000, dim = 300)

#sample1 = bpemb_vi.encode("Sơn Đinh một hai ba bốn năm")
#print(sample1)

data = pd.read_csv('data_qipdec.csv', delimiter=',')
data = data.drop(columns=['Url'])
#print(data.head(10))

dict = {}
#print(data)
# Nhúng từng từ trong tập dữ liệu
for i in tqdm(data['Words']):
    ids  = bpemb_vi.encode_ids(i)
    embedding = bpemb_vi.vectors[ids]
    embedding = np.mean(embedding, axis=0)
    # print(embedding.shape)
    # Tính embed cho từng từ
    dict[i] = embedding
    #print(i)

#print(dict)

import pickle
with open('dict.pickle', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('dict.pickle', 'rb') as f:
    loaded = pickle.load(f)
    # print(loaded)