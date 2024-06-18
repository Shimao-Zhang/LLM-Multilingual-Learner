import sys
# sys.path.append('/home/nfs03/zhangsm/multiL-transfer-interpretability/llm-latent-language')

import pandas as pd
import os
from dataclasses import dataclass
import json
import time
import random
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from llamawrapper import load_unemb_only, LlamaHelper
import seaborn as sns
from scipy.stats import bootstrap
from utils import plot_ci, plot_ci_plus_heatmap
from tqdm import tqdm
from datasets import load_dataset, load_metric
from typing import Dict, List, Tuple
import torch.nn.functional as Func
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

def set_seed(seed: int = 719):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True

class CustomDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 根据实际情况处理数据
        sample = self.data[idx]
        return sample

# fix random seed
seed = 0
set_seed(seed=seed)

# 路径语言

model_size = 'mistral-7b'

target_lang = 'ru'

print(f"\nStart loading model {model_size}...\n")

all_avaliable_model = {'mistral-7b':'/home/nfs02/model/mistralai_Mistral-7B-v0.1',
                       'mistral-7b-aligned':'/home/nfs03/zhangsm/multiL-transfer-interpretability/pretrained-models/mistral_zhit20k_round1_epoch3',
                       'Qwen1.5-0.5b':'/home/nfs02/model/Qwen1.5-0.5B',
                       'Qwen1.5-1.8b':'/home/nfs02/model/Qwen1.5-1.8B',
                       'Qwen1.5-1.8b-aligned':'/home/nfs03/zhangsm/multiL-transfer-interpretability/pretrained-models/Qwen1.8b_emotion_zhde20k_round1_epoch3',
                       'Qwen1.5-4b':'/home/nfs02/model/Qwen1.5-4B',
                       'Qwen1.5-4b-aligned':'/home/nfs03/zhangsm/multiL-transfer-interpretability/pretrained-models/Qwen4b_emotion_swhi20k_round1_epoch3',
                       'Qwen1.5-14b':'/home/nfs02/model/Qwen1.5-14B-Base',
                       'Qwen1.5-14b-aligned':'/home/nfs03/zhangsm/multiL-transfer-interpretability/pretrained-models/Qwen14b_emotion_swhi20k_round1_epoch3'}
custom_model = all_avaliable_model[model_size]
single_token_only = False
multi_token_only = False
out_dir = '../zsm-results'
lang2name = {'fr': 'Français', 'de': 'Deutsch', 'ru': 'Русский', 'en': 'English', 'zh': '中文'}

if single_token_only and multi_token_only:
    raise ValueError('single_token_only and multi_token_only cannot be True at the same time')

# unemb = load_unemb_only(model_size)
llama = LlamaHelper(dir=custom_model, load_in_4bit=True, device_map='auto')
tokenizer = llama.tokenizer
model = llama.model
unemb = nn.Sequential(llama.model.model.norm, llama.model.lm_head)
print(unemb)
id2voc = {id:voc for voc, id in llama.tokenizer.get_vocab().items()}
voc2id = llama.tokenizer.get_vocab()


print(f"\nStart loading {target_lang} dataset...\n")

if target_lang == 'en':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_en500_test.json')
elif target_lang == 'zh':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_zh500_test.json')
elif target_lang == 'ja':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_ja500_test.json')
elif target_lang == 'hi':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_hi500_test.json')
elif target_lang == 'sw':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_sw500_test.json')
elif target_lang == 'bn':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_bn500_test.json')
elif target_lang == 'it':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_it500_test.json')
elif target_lang == 'no':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_no500_test.json')
elif target_lang == 'es':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_es500_test.json')
elif target_lang == 'is':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_is500_test.json')
elif target_lang == 'bg':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_bg500_test.json')
elif target_lang == 'sv':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_sv500_test.json')
elif target_lang == 'sl':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_sl500_test.json')
elif target_lang == 'ms':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_ms500_test.json')
elif target_lang == 'th':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_th500_test.json')
elif target_lang == 'pl':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_pl500_test.json')
elif target_lang == 'ru':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_ru500_test.json')
elif target_lang == 'de':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_de500_test.json')
elif target_lang == 'fr':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_fr500_test.json')
elif target_lang == 'nl':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/ap_emotion/emotion_nl500_test.json')

assert(dataset is not None)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
sample_iterator = tqdm(dataloader, desc="Processed batchs num")


def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n+1)]
    return tokens 

def add_spaces(tokens):
    return ['▁' + t for t in tokens] + tokens

def capitalizations(tokens):
    return list(set(tokens))

def unicode_prefix_tokid(zh_char = "积", tokenizer=tokenizer):
    if not zh_char.encode().__str__()[2:-1].startswith('\\x'):
        return None
    start = zh_char.encode().__str__()[2:-1].split('\\x')[1]
    unicode_format = '<0x%s>'
    start_key = unicode_format%start.upper()
    if start_key in tokenizer.get_vocab():
        return tokenizer.get_vocab()[start_key]
    return None

def process_tokens(token_str: str, tokenizer):
    with_prefixes = token_prefixes(token_str)
    # print("with_prefixes: ", with_prefixes)
    with_spaces = add_spaces(with_prefixes)
    with_capitalizations = capitalizations(with_spaces)
    final_tokens = []
    for tok in with_capitalizations:
        if tok in tokenizer.get_vocab():
            final_tokens.append(tokenizer.get_vocab()[tok])
    # if lang in ['zh', 'ja', 'hi', 'ru', 'th', 'bn']:
    #     tokid = unicode_prefix_tokid(token_str[0], tokenizer)
    #     if tokid is not None:
    #         final_tokens.append(tokid)
    tokid = unicode_prefix_tokid(token_str[0], tokenizer)
    if tokid is not None:
        final_tokens.append(tokid)
    
    if "Qwen" in model_size:
        with_prefixes = token_prefixes(token_str)
        with_capitalizations = capitalizations(with_prefixes)
        for tok in with_capitalizations:
            id = tokenizer(tok, return_tensors='pt')['input_ids'][0][0].item()
            final_tokens.append(id)
    
    final_tokens = list(set(final_tokens))
    return final_tokens

def get_tokens(token_ids, id2voc=id2voc):
    return [id2voc[tokid] for tokid in token_ids]

def compute_entropy(probas):
    return (-probas*torch.log2(probas)).sum(dim=-1)


# prepare for energy plots
U = list(unemb[1].parameters())[0].detach().cpu().float()
weights = list(unemb[0].parameters())[0].detach().cpu().float()
print(f'U {U.shape} weights {weights.unsqueeze(0).shape}')
U_weighted = U.clone() 
#U_weighted = U_weighted / ((U_weighted**2).mean(dim=1, keepdim=True))**0.5
U_weighted *= weights.unsqueeze(0)
U_normalized = U_weighted / ((U_weighted**2).sum(dim=1, keepdim=True))**0.5
v = U.shape[0]
TT = U_normalized.T @ U_normalized
avgUU = (((U_normalized.T @ U_normalized)**2).sum() / v**2)**0.5
print(avgUU.item())


# 转化输入的temple，获取检测的最终内容和中间内容的token
def changeInupt(n_skip = 2):
    dataset_gap = []
    key = "blank_prompt_translation_masked"
    for step, inputs in enumerate(sample_iterator):
        # get tok sets and kick out if intersection
        out_token_str = inputs['answer'][0]
        # print("out_token_str: ", out_token_str)
        if (out_token_str == 'positive' or out_token_str == '积极' or out_token_str == 'ポジティブ' or out_token_str == 'सकारात्मक' 
            or out_token_str == 'chanya' or out_token_str == 'ইতিবাচক' or out_token_str == 'deimhinneach'
        or out_token_str == 'positivo' or out_token_str == 'positivt' or out_token_str == 'позитивный' or out_token_str == 'jákvæð' 
        or out_token_str == 'positiv' or out_token_str == 'pozitivno' or out_token_str == 'positif' 
        or out_token_str == 'เชิงบวก' or out_token_str == 'ਸਕਾਰਾਤਮਕ'or out_token_str == 'pozytywny'):
            latent_token_str = 'positive'
        elif (out_token_str == 'negative' or out_token_str == '消极' or out_token_str == 'ネガティブ' or out_token_str == 'नकारात्मक' 
              or out_token_str == 'hasi' or out_token_str == 'নেতিবাচক' or out_token_str == 'àicheil'
              or out_token_str == 'negativo' or out_token_str == 'negativ' or out_token_str == 'отрицательный' or out_token_str == 'neikvæð'
              or out_token_str == 'negativ' or out_token_str == 'negativno' or out_token_str == 'negatif'
              or out_token_str == 'เชิงลบ' or out_token_str == 'ਨਕਾਰਾਤਮਕ'or out_token_str == 'negatywny'):
            latent_token_str = 'negative'
        out_token_id = process_tokens(out_token_str, tokenizer)
        latent_token_id = process_tokens(latent_token_str, tokenizer)
        intersection = set(out_token_id).intersection(set(latent_token_id))
        if len(out_token_id) == 0 or len(latent_token_id) == 0:
            continue
        if target_lang != 'en' and len(intersection) > 0:
            continue
        
        dataset_gap.append({
            'prompt': inputs['input'],
            'out_token_id': out_token_id,
            'out_token_str': out_token_str,
            'latent_token_id': latent_token_id,
            'latent_token_str': latent_token_str,
        })
    return dataset_gap


dataset_gap = changeInupt()
print("Dataset Length: ", len(dataset_gap))

all_possible_latent_tokens_id = list(set(process_tokens("positive", tokenizer) + process_tokens("negative", tokenizer)))
if target_lang == 'en':
    all_possible_out_token_id = list(set(process_tokens("positive", tokenizer) + process_tokens("negative", tokenizer)))
elif target_lang == 'zh':
    all_possible_out_token_id = list(set(process_tokens("积极", tokenizer) + process_tokens("消极", tokenizer)))
elif target_lang == 'ja':
    all_possible_out_token_id = list(set(process_tokens("ポジティブ", tokenizer) + process_tokens("ネガティブ", tokenizer)))
elif target_lang == 'hi':
    all_possible_out_token_id = list(set(process_tokens("सकारात्मक", tokenizer) + process_tokens("नकारात्मक", tokenizer)))
elif target_lang == 'sw':
    all_possible_out_token_id = list(set(process_tokens("chanya", tokenizer) + process_tokens("hasi", tokenizer)))
elif target_lang == 'bn':
    all_possible_out_token_id = list(set(process_tokens("ইতিবাচক", tokenizer) + process_tokens("নেতিবাচক", tokenizer)))
elif target_lang == 'it':
    all_possible_out_token_id = list(set(process_tokens("positivo", tokenizer) + process_tokens("negativo", tokenizer)))
elif target_lang == 'no':
    all_possible_out_token_id = list(set(process_tokens("positivt", tokenizer) + process_tokens("negativ", tokenizer)))
elif target_lang == 'es':
    all_possible_out_token_id = list(set(process_tokens("positivo", tokenizer) + process_tokens("negativo", tokenizer)))
elif target_lang == 'is':
    all_possible_out_token_id = list(set(process_tokens("jákvæð", tokenizer) + process_tokens("neikvæð", tokenizer)))
elif target_lang == 'bg':
    all_possible_out_token_id = list(set(process_tokens("положителен", tokenizer) + process_tokens("отрицателен", tokenizer)))
elif target_lang == 'sv':
    all_possible_out_token_id = list(set(process_tokens("positiv", tokenizer) + process_tokens("negativ", tokenizer)))
elif target_lang == 'sl':
    all_possible_out_token_id = list(set(process_tokens("pozitivno", tokenizer) + process_tokens("negativno", tokenizer)))
elif target_lang == 'ms':
    all_possible_out_token_id = list(set(process_tokens("positif", tokenizer) + process_tokens("negatif", tokenizer)))
elif target_lang == 'th':
    all_possible_out_token_id = list(set(process_tokens("เชิงบวก", tokenizer) + process_tokens("เชิงลบ", tokenizer)))
elif target_lang == 'ru':
    all_possible_out_token_id = list(set(process_tokens("позитивный", tokenizer) + process_tokens("отрицательный", tokenizer)))

df_gap = pd.DataFrame(dataset_gap)
print(df_gap['prompt'][0])


os.makedirs(f'{os.path.join(out_dir, model_size)}/emotion', exist_ok=True)
df_gap.to_csv(f'{os.path.join(out_dir, model_size)}/emotion/{target_lang}_dataset.csv', index=False)


latent_token_probs = []
out_token_probs = []
all_possible_latent_token_probs = []
all_possible_out_token_probs = []
entropy = []
energy = [] 
latents_all = []


print("dataset_gap: ", dataset_gap[0])

for idx, d in enumerate(tqdm(dataset_gap, desc="Inference instance num")):

    prompt_prefix = {'en': '''It starts off a bit slow, but once the product placement jokes start it takes off.
Emotion: "positive"             
I've read this book with much expectation, it was very boring all through out the book
Emotion: "negative"''',
                 'zh': '''一开始有点慢，但一旦植入式笑话开始，它就会起飞。
情感: "积极"
我带着很大的期待读了这本书，整本书都很无聊
情感: "消极"''',
                 'ja': '''始まりは少し遅いですが、プロダクト プレイスメントのジョークが始まると一気に盛り上がります。
感情: "ポジティブ"
とても期待してこの本を読んだのですが、全編通してとても退屈でした
感情: "ネガティブ"''',
                 'hi': '''इसकी शुरुआत थोड़ी धीमी होती है, लेकिन एक बार जब उत्पाद प्लेसमेंट के बारे में मजाक शुरू हो जाता है तो इसमें तेजी आ जाती है।
भावना: "सकारात्मक"
मैंने यह किताब बहुत उम्मीद के साथ पढ़ी है, पूरी किताब में यह बहुत उबाऊ थी
भावना: "नकारात्मक"''',
                  'sw':'''Huanza polepole, lakini utani wa uwekaji bidhaa unapoanza huanza.
Hisia: "chanya"
Nimekisoma kitabu hiki kwa matarajio mengi, kilikuwa cha kuchosha katika kitabu chote
Hisia: "hasi"''',
                  'bn':'''এটি একটু ধীর গতিতে শুরু হয়, কিন্তু একবার পণ্য প্লেসমেন্ট জোকস শুরু হলে এটি বন্ধ হয়ে যায়।
আবেগ: "ইতিবাচক"
আমি অনেক প্রত্যাশা নিয়ে এই বইটি পড়েছি, বইটি জুড়ে এটি খুব বিরক্তিকর ছিল
আবেগ: "নেতিবাচক"''',
                  'it':'''Inizia un po' lentamente, ma una volta che iniziano le battute sul posizionamento del prodotto, decolla.
Emozione: "positivo"
Ho letto questo libro con molte aspettative, è stato molto noioso durante tutto il libro
Emozione: "negativo"''',
                  'no':'''Det starter litt tregt, men når produktplasseringsvitsene starter tar det av.
Følelse: "positivt"
Jeg har lest denne boken med store forventninger, den var veldig kjedelig gjennom hele boken
Følelse: "negativ"''',
                  'es':'''Comienza un poco lento, pero una vez que comienzan los chistes sobre la colocación de productos, despega.
Emoción: "positivo"
He leído este libro con muchas expectativas, fue muy aburrido durante todo el libro.
Emoción: "negativo"''',
                  'is':'''Það byrjar svolítið hægt, en þegar vörustaðsetningarbrandararnir byrja þá tekur það af.
Tilfinning: "jákvæð"
Ég hef lesið þessa bók með mikilli eftirvæntingu, hún var mjög leiðinleg alla bókina
Tilfinning: "neikvæð"''',
                  'bg':'''Започва малко бавно, но след като започнат шегите за позициониране на продукти, започва.
Емоция: "положителен"
Прочетох тази книга с много очаквания, беше много скучна през цялата книга
Емоция: "отрицателен"''',
                  'sv':'''Det börjar lite långsamt, men när produktplaceringsskämten väl börjar så tar det fart.
Känsla: "positiv"
Jag har läst den här boken med stora förväntningar, den var väldigt tråkig genom hela boken
Känsla: "negativ"''',
                  'sl':'''Začne se nekoliko počasi, a ko se začnejo šale o promocijskem prikazovanju izdelkov, začne.
Čustvo: "pozitivno"
To knjigo sem prebrala z velikim pričakovanjem, vsa knjiga je bila zelo dolgočasna
Čustvo: "negativno"''',
                  'ms':'''Ia bermula agak perlahan, tetapi apabila jenaka peletakan produk bermula, ia bermula.
Emosi: "positif"
Saya telah membaca buku ini dengan penuh harapan, ia sangat membosankan sepanjang buku itu
Emosi: "negatif"''',
                  'th':'''มันเริ่มต้นช้านิดหน่อย แต่เมื่อเรื่องตลกเกี่ยวกับการจัดวางผลิตภัณฑ์เริ่มต้นขึ้น
อารมณ์: "เชิงบวก"
ฉันอ่านหนังสือเล่มนี้ด้วยความคาดหวังมาก มันน่าเบื่อมากตลอดทั้งเล่ม
อารมณ์: "เชิงลบ"''',
                  'ru':'''Все начинается немного медленно, но как только начинаются шутки о продакт-плейсменте, все начинается.
Эмоция: "позитивный"
Я читала эту книгу с большим нетерпением, она была очень скучной на протяжении всей книги.
Эмоция: "отрицательный"'''}
    # print(prompt)
    if target_lang == 'en':
        prompt = [prompt_prefix['en'] + '\n' + i + '\nEmotion: \"' for i in d['prompt']]
    elif target_lang == 'zh':
        prompt = [prompt_prefix['zh'] + '\n' + i + '\n情感: \"' for i in d['prompt']]
    elif target_lang == 'ja':
        prompt = [prompt_prefix['ja'] + '\n' + i + '\n感情: \"' for i in d['prompt']]
    elif target_lang == 'hi':
        prompt = [prompt_prefix['hi'] + '\n' + i + '\nभावना: \"' for i in d['prompt']]
    elif target_lang == 'sw':
        prompt = [prompt_prefix['sw'] + '\n' + i + '\nHisia: \"' for i in d['prompt']]
    elif target_lang == 'bn':
        prompt = [prompt_prefix['bn'] + '\n' + i + '\nআবেগ: \"' for i in d['prompt']]
    elif target_lang == 'it':
        prompt = [prompt_prefix['it'] + '\n' + i + '\nEmozione: \"' for i in d['prompt']]
    elif target_lang == 'no':
        prompt = [prompt_prefix['no'] + '\n' + i + '\nFølelse: \"' for i in d['prompt']]
    elif target_lang == 'es':
        prompt = [prompt_prefix['es'] + '\n' + i + '\nEmoción: \"' for i in d['prompt']]
    elif target_lang == 'is':
        prompt = [prompt_prefix['is'] + '\n' + i + '\nTilfinning: \"' for i in d['prompt']]
    elif target_lang == 'bg':
        prompt = [prompt_prefix['bg'] + '\n' + i + '\nЕмоция: \"' for i in d['prompt']]
    elif target_lang == 'sv':
        prompt = [prompt_prefix['sv'] + '\n' + i + '\nKänsla: \"' for i in d['prompt']]
    elif target_lang == 'sl':
        prompt = [prompt_prefix['sl'] + '\n' + i + '\nČustvo: \"' for i in d['prompt']]
    elif target_lang == 'ms':
        prompt = [prompt_prefix['ms'] + '\n' + i + '\nEmosi: \"' for i in d['prompt']]
    elif target_lang == 'th':
        prompt = [prompt_prefix['th'] + '\n' + i + '\nอารมณ์: \"' for i in d['prompt']]
    elif target_lang == 'ru':
        prompt = [prompt_prefix['ru'] + '\n' + i + '\nЭмоция: \"' for i in d['prompt']]
    
    latents = llama.latents_all_layers(prompt)
    latents = latents.to('cuda')
    logits = unemb(latents)
    # print("logits shape: ", logits.shape)
    last = logits[:, -1, :].float().softmax(dim=-1).detach().cpu()
    # print("last logits: ", last)
    # print("last logits shape: ", last.shape)
    latent_token_probs += [last[:, torch.tensor(d['latent_token_id'])].sum(axis=-1)]
    out_token_probs += [last[:, torch.tensor(d['out_token_id'])].sum(axis=-1)]
    all_possible_latent_token_probs += [last[:, torch.tensor(all_possible_latent_tokens_id)].sum(axis=-1)]
    all_possible_out_token_probs += [last[:, torch.tensor(all_possible_out_token_id)].sum(axis=-1)]
    max_probs_idx = [torch.argmax(layer_probs) for layer_probs in last]
    # print([idx.item() for idx in max_probs_idx])
    # print([id2voc[idx.item()] for idx in max_probs_idx])
    # print("latent_token_probs: ", latent_token_probs)
    # print("out_token_probs: ", out_token_probs)
    # print("all_possible_latent_token_probs: ", all_possible_latent_token_probs)
    # print("all_possible_out_token_probs: ", all_possible_out_token_probs)
    entropy += [compute_entropy(last)]
    latents_all += [latents[:, -1, :].float().detach().cpu().clone()]
    latents_normalized = latents[:, -1, :].float().detach().cpu().clone()
    latents_normalized = latents_normalized / (((latents_normalized**2).mean(dim=-1, keepdim=True))**0.5)
    latents_normalized /= (latents_normalized.norm(dim=-1, keepdim=True))
    norm = ((U_normalized @ latents_normalized.T)**2).mean(dim=0)**0.5
    energy += [norm/avgUU]

latent_token_probs = torch.stack(latent_token_probs)
out_token_probs = torch.stack(out_token_probs)
all_possible_latent_token_probs = torch.stack(all_possible_latent_token_probs)
all_possible_out_token_probs = torch.stack(all_possible_out_token_probs)
entropy = torch.stack(entropy)
energy = torch.stack(energy)
latents = torch.stack(latents_all)

print(f"Using the model in {custom_model}")


# 画图
size2tik = {'llama2-7b': 5, 'llama2-7b-aligned': 5, 'llama2-13b': 5, 'mistral-7b': 5, 'mistral-7b-aligned': 5, 'Qwen1.5-1.8b':5, 'Qwen1.5-1.8b-aligned':5}

fig, ax, ax2 = plot_ci_plus_heatmap(latent_token_probs, entropy, 'en', color='tab:green', tik_step=size2tik[model_size], do_colorbar=True,
nums=[.99, 0.18, 0.025, 0.6])
if target_lang != 'en':
    plot_ci(ax2, out_token_probs, target_lang, color='tab:orange', do_lines=False)
    plot_ci(ax2, all_possible_out_token_probs, 'all_possible_out', color='tab:blue', do_lines=False)
    plot_ci(ax2, all_possible_latent_token_probs, 'all_possible_latent', color='tab:red', do_lines=False)
else:
    plot_ci(ax2, all_possible_latent_token_probs, 'all_possible_latent', color='tab:red', do_lines=True)
ax2.set_xlabel('layer')
ax2.set_ylabel('probability')
ax2.set_xlim(0, out_token_probs.shape[1]+1)
ax2.set_ylim(0, 1)
# put legend on the top left
ax2.legend(loc='upper left', fontsize=14)
# plt.tight_layout()
os.makedirs(f'{os.path.join(out_dir, model_size)}/emotion/2-shot/figures', exist_ok=True)
plt.savefig(f'{os.path.join(out_dir, model_size)}/emotion/2-shot/figures/{model_size}_{target_lang}_probas_ent.pdf', dpi=300, bbox_inches='tight')


# fig, ax2 = plt.subplots(figsize=(5,3))
# plot_ci(ax2, energy, 'energy', color='tab:green', do_lines=True, tik_step=size2tik[model_size])
# ax2.set_xlabel('layer')
# ax2.set_ylabel('energy')
# if model_size == '7b':
#     ax2.set_xlim(0, out_token_probs.shape[1]+1)
# else:
#     ax2.set_xlim(0, round(out_token_probs.shape[1]/10)*10+1)
# os.makedirs(f'{os.path.join(out_dir, custom_model)}/cloze', exist_ok=True)
# plt.savefig(f'{os.path.join(out_dir, custom_model)}/cloze/{model_size}_{target_lang}_energy.pdf', dpi=300, bbox_inches='tight')