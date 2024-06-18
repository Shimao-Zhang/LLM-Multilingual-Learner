import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from safetensors.torch import load_file
from collections import defaultdict
import numpy as np
import torch
import os
import random
import time
import json
from tqdm import tqdm, trange
from datasets import load_dataset, load_metric
from typing import Dict, List, Tuple
import torch.nn.functional as Func
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BitsAndBytesConfig

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

model_size = 'Qwen1.5-14b-aligned'

target_lang = 'bn'

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
model_name_or_path = all_avaliable_model[model_size]

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=None)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_auth_token=None, device_map='auto', quantization_config=quantization_config)

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=None)
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_auth_token=None, torch_dtype='auto', device_map='auto')
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_auth_token=None, device_map='auto')

model.eval()

tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

id2voc = {id:voc for voc, id in tokenizer.get_vocab().items()}
voc2id = tokenizer.get_vocab()


def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n+1)]
    return tokens 

def add_spaces(tokens):
    return ['▁' + t for t in tokens] + tokens

def capitalizations(tokens):
    return list(set(tokens))

def unicode_prefix_tokid(zh_char = "蕴", tokenizer=tokenizer):
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
    # if lang in ['zh', 'ja', 'hi', 'ru', 'th', 'bn', 'gd', 'ay']:
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



print(f"\nStart loading {target_lang} dataset...\n")

if target_lang == 'en':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_en500_test.json')
elif target_lang == 'zh':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_zh500_test.json')
elif target_lang == 'ja':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_ja500_test.json')
elif target_lang == 'hi':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_hi500_test.json')
elif target_lang == 'sw':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_sw500_test.json')
elif target_lang == 'bn':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_bn500_test.json')
elif target_lang == 'it':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_it500_test.json')
elif target_lang == 'no':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_no500_test.json')
elif target_lang == 'es':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_es500_test.json')
elif target_lang == 'is':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_is500_test.json')
elif target_lang == 'bg':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_bg500_test.json')
elif target_lang == 'sv':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_sv500_test.json')
elif target_lang == 'sl':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_sl500_test.json')
elif target_lang == 'ms':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_ms500_test.json')
elif target_lang == 'th':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_th500_test.json')
elif target_lang == 'pl':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_pl500_test.json')
elif target_lang == 'ru':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_ru500_test.json')
elif target_lang == 'de':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_de500_test.json')
elif target_lang == 'fr':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_fr500_test.json')
elif target_lang == 'nl':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/paws/paws_nl500_test.json')

assert(dataset is not None)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
sample_iterator = tqdm(dataloader, desc="Inferenced batchs num")

prompt_prefix = {'en': '''Sentence 1: The 2005 North Indian Ocean cyclone season was weak to southern India despite the destructive and deadly storms .
Sentence 2: The North Indian Ocean cyclone season in 2005 , despite the destructive and lethal storms , was weak to southern India .
Answer: "same"
Sentence 1: Saunders defeated Dan Barrera at by unanimous decision .
Sentence 2: By unanimous decision Dan Barrera defeated Saunders .
Answer: "different"
Sentence 1: Match 5 : Yoji Anjo defeats Kazushi Sakuraba ( submission )
Sentence 2: Match 5 : Yoji Anjo defeats kazushi Sakuraba ( leg submission )
Answer: "same"
Sentence 1: He decided to teach Indian history `` because he wanted to study it '' .
Sentence 2: He decided to study Indian history because he wanted to teach it .
Answer: "different"''',
                 'zh': '''句子 1: 2005 年北印度洋气旋季节尽管造成了破坏性和致命的风暴，但对印度南部的影响仍然较弱。
句子 2: 2005年的北印度洋气旋季节，尽管有破坏性和致命性的风暴，但对印度南部的影响却很弱。
答案: "same"
句子 1: 桑德斯一致判定击败丹·巴雷拉 (Dan Barrera)。
句子 2: 一致决定丹·巴雷拉 (Dan Barrera) 击败了桑德斯 (Saunders)。
答案: "different"
句子 1: 第 5 场比赛：安城洋次击败樱庭一志（降服）。
句子 2: 第 5 场比赛：安城洋次击败樱庭一志（腿部降服）。
答案: "same"
句子 1: 他决定教授印度历史“因为他想研究它”。
句子 2: 他决定学习印度历史，因为他想教授印度历史。
答案: "different"''',
                 'ja': '''文 1: 2005 年の北インド洋サイクロンシーズンは、破壊的で致命的な嵐にもかかわらず、南インドに対して弱かった。
文 2: 2005 年の北インド洋サイクロンシーズンは、破壊的で致死的な嵐にもかかわらず、南インドに対しては弱かった。
答え: "same"
文 1: サンダースは全会一致の決定でダン・バレラを破った。
文 2: 全会一致の決定により、ダン バレラはサンダースを破りました。
答え: "different"
文 1: 試合 5 : 安城洋司が桜庭和志に勝つ (サブミッション)
文 2: 試合 5 : 安城洋司が桜庭和志に勝つ (レッグサブミッション)
答え: "same"
文 1: 彼はインドの歴史を「勉強したかったから」教えることにした。
文 2: 彼はインドの歴史を教えたかったので、インドの歴史を学ぶことにしました。
答え: "different"''',
                 'hi': '''वाक्य 1: विनाशकारी और जानलेवा तूफानों के बावजूद 2005 का उत्तरी हिंद महासागर चक्रवात का मौसम दक्षिणी भारत के लिए कमज़ोर था।
वाक्य 2: विनाशकारी और जानलेवा तूफानों के बावजूद 2005 का उत्तरी हिंद महासागर चक्रवात का मौसम दक्षिणी भारत के लिए कमज़ोर था।
उत्तर: "same"
वाक्य 1: सॉन्डर्स ने सर्वसम्मति से डैन बैरेरा को हराया।
वाक्य 2: सर्वसम्मति से डैन बैरेरा ने सॉन्डर्स को हराया।
उत्तर: "different"
वाक्य 1: मैच 5: योजी अंजो ने काज़ुशी सकुराबा को हराया (सबमिशन)
वाक्य 2: मैच 5: योजी अंजो ने काज़ुशी सकुराबा को हराया (लेग सबमिशन)
उत्तर: "same"
वाक्य 1: उन्होंने भारतीय इतिहास पढ़ाने का फैसला किया "क्योंकि वे इसका अध्ययन करना चाहते थे"।
वाक्य 2: उन्होंने भारतीय इतिहास पढ़ाने का फैसला किया क्योंकि वे इसका अध्ययन करना चाहते थे।
उत्तर: "different"''',
                 'sw':'''Sentensi 1: Msimu wa kimbunga wa Bahari ya Hindi Kaskazini wa 2005 ulikuwa dhaifu hadi kusini mwa India licha ya dhoruba mbaya na mbaya.
Sentensi 2: Msimu wa kimbunga wa Bahari ya Hindi Kaskazini mwaka 2005, licha ya dhoruba haribifu na mbaya, ulikuwa dhaifu hadi kusini mwa India.
Jibu: "same"
Sentensi 1: Saunders alimshinda Dan Barrera kwa uamuzi wa kauli moja.
Sentensi 2: Kwa uamuzi wa pamoja Dan Barrera alimshinda Saunders.
Jibu: "different"
Sentensi 1: Mechi ya 5: Yoji Anjo amshinda Kazushi Sakuraba ( kuwasilisha )
Sentensi 2: Mechi ya 5: Yoji Anjo amshinda Kazushi Sakuraba ( kuwasilisha mguu)
Jibu: "same"
Sentensi 1: Aliamua kufundisha historia ya Kihindi "kwa sababu alitaka kuisoma" .
Sentensi 2: Aliamua kusoma historia ya India kwa sababu alitaka kuifundisha.
Jibu: "different"''',
                 'bn':'''বাক্য 1: 2005 সালের উত্তর ভারত মহাসাগরের ঘূর্ণিঝড়ের মরসুম ধ্বংসাত্মক এবং মারাত্মক ঝড় সত্ত্বেও দক্ষিণ ভারতের কাছে দুর্বল ছিল।
বাক্য 2: 2005 সালে উত্তর ভারত মহাসাগরের ঘূর্ণিঝড়ের মরসুম, ধ্বংসাত্মক এবং প্রাণঘাতী ঝড় সত্ত্বেও, দক্ষিণ ভারতের প্রতি দুর্বল ছিল।
উত্তর: "same"
বাক্য 1: সন্ডার্স সর্বসম্মত সিদ্ধান্তে ড্যান ব্যারেরাকে পরাজিত করেন।
বাক্য 2: সর্বসম্মত সিদ্ধান্তে ড্যান ব্যারেরা সন্ডার্সকে পরাজিত করেন।
উত্তর: "different"
বাক্য 1: ম্যাচ 5 : ইয়োজি আনজো কাজুশি সাকুরাবাকে পরাজিত করেছেন (জমা)
বাক্য 2: ম্যাচ 5 : ইয়োজি আনজো কাজুশি সাকুরাবাকে পরাজিত করেছেন (পা জমা)
উত্তর: "same"
বাক্য 1: তিনি ভারতীয় ইতিহাস শেখানোর সিদ্ধান্ত নিয়েছিলেন "কারণ তিনি এটি অধ্যয়ন করতে চেয়েছিলেন"।
বাক্য 2: তিনি ভারতীয় ইতিহাস অধ্যয়ন করার সিদ্ধান্ত নিয়েছিলেন কারণ তিনি এটি শেখাতে চেয়েছিলেন।
উত্তর: "different"''',
                 'it':'''Frase 1: La stagione dei cicloni del 2005 nell'Oceano Indiano settentrionale è stata debole nell'India meridionale, nonostante le tempeste distruttive e mortali.
Frase 2: La stagione dei cicloni nell'Oceano Indiano settentrionale nel 2005, nonostante le tempeste distruttive e letali, è stata debole nell'India meridionale.
Risposta: "same"
Frase 1: Saunders ha sconfitto Dan Barrera con decisione unanime.
Frase 2: Con decisione unanime Dan Barrera ha sconfitto Saunders.
Risposta: "different"
Frase 1: Match 5: Yoji Anjo sconfigge Kazushi Sakuraba (sottomissione)
Frase 2: Match 5: Yoji Anjo sconfigge kazushi Sakuraba (sottomissione alle gambe)
Risposta: "same"
Frase 1: Ha deciso di insegnare la storia indiana "perché voleva studiarla".
Frase 2: Ha deciso di studiare la storia indiana perché voleva insegnarla.
Risposta: "different"''',
                 'no':'''Setning 1: Syklonsesongen i Nord-Indiahavet i 2005 var svak for Sør-India til tross for de destruktive og dødelige stormene.
Setning 2: Syklonsesongen i Nord-Indiahavet i 2005, til tross for de destruktive og dødelige stormene, var svak for Sør-India.
Svar: "same"
Setning 1: Saunders beseiret Dan Barrera ved enstemmig avgjørelse.
Setning 2: Ved enstemmig avgjørelse beseiret Dan Barrera Saunders.
Svar: "different"
Setning 1: Kamp 5: Yoji Anjo beseirer Kazushi Sakuraba (submission)
Setning 2: Kamp 5: Yoji Anjo beseirer kazushi Sakuraba (beninnlevering)
Svar: "same"
Setning 1: Han bestemte seg for å undervise i indisk historie "fordi han ønsket å studere det".
Setning 2: Han bestemte seg for å studere indisk historie fordi han ønsket å undervise i det.
Svar: "different"''',
                 'es':'''Sentencia 1: La temporada de ciclones del norte del Océano Índico de 2005 fue débil para el sur de la India a pesar de las tormentas destructivas y mortales.
Sentencia 2: La temporada de ciclones del norte del Océano Índico en 2005, a pesar de las tormentas destructivas y letales, fue débil para el sur de la India.
Respuesta: "same"
Sentencia 1: Saunders derrotó a Dan Barrera por decisión unánime.
Sentencia 2: Por decisión unánime Dan Barrera derrotó a Saunders.
Respuesta: "different"
Sentencia 1: Partido 5: Yoji Anjo derrota a Kazushi Sakuraba (sumisión)
Sentencia 2: Partido 5: Yoji Anjo derrota a Kazushi Sakuraba (sumisión de pierna)
Respuesta: "same"
Sentencia 1: Decidió enseñar historia de la India "porque quería estudiarla".
Sentencia 2: Decidió estudiar historia de la India porque quería enseñarla.
Respuesta: "different"''',
                 'is':'''Setning 1: Hvirfilbylgjutímabilið í Norður-Indlandshafi 2005 var veikt fyrir suðurhluta Indlands þrátt fyrir eyðileggjandi og banvæna storma.
Setning 2: Hvirfilbylgjutímabilið í Norður-Indlandshafi árið 2005, þrátt fyrir eyðileggjandi og banvæna storma, var veikburða í suðurhluta Indlands.
Svaraðu: "same"
Setning 1: Saunders sigraði Dan Barrera eftir einróma dómaraákvörðun.
Setning 2: Með einróma ákvörðun sigraði Dan Barrera Saunders.
Svaraðu: "different"
Setning 1: Leikur 5: Yoji Anjo sigrar Kazushi Sakuraba (uppgjöf)
Setning 2: Leikur 5: Yoji Anjo sigrar kazushi Sakuraba (fótauppgjöf)
Svaraðu: "same"
Setning 1: Hann ákvað að kenna indverska sögu "vegna þess að hann vildi læra hana".
Setning 2: Hann ákvað að læra indverska sögu vegna þess að hann vildi kenna hana.
Svaraðu: "different"''',
                 'bg':'''Изречение 1: Сезонът на циклона в Северен Индийски океан през 2005 г. беше слаб за Южна Индия въпреки разрушителните и смъртоносни бури.
Изречение 2: Сезонът на циклона в Северния Индийски океан през 2005 г., въпреки разрушителните и смъртоносни бури, беше слаб за южна Индия.
Отговор: "same"
Изречение 1: Сондърс победи Дан Барера на с единодушно съдийско решение.
Изречение 2: С единодушно съдийско решение Dan Barrera победи Saunders.
Отговор: "different"
Изречение 1: Мач 5: Йоджи Анджо побеждава kazushi Sakuraba (събмишън)
Изречение 2: Мач 5: Йоджи Анджо побеждава kazushi Sakuraba (субмишън с крак)
Отговор: "same"
Изречение 1: Той реши да преподава индийска история, „защото искаше да я изучава“.
Изречение 2: Той реши да изучава индийска история, защото искаше да я преподава.
Отговор: "different"''',
                 'sv':'''Mening 1: Cyklonsäsongen 2005 i norra Indiska oceanen var svag för södra Indien trots de destruktiva och dödliga stormarna.
Mening 2: Cyklonsäsongen i norra Indiska oceanen 2005, trots de destruktiva och dödliga stormarna, var svag för södra Indien.
Svar: "same"
Mening 1: Saunders besegrade Dan Barrera med enhälligt beslut.
Mening 2: Genom enhälligt beslut besegrade Dan Barrera Saunders.
Svar: "different"
Mening 1: Match 5: Yoji Anjo besegrar Kazushi Sakuraba (submission)
Mening 2: Match 5: Yoji Anjo besegrar kazushi Sakuraba (ben submission)
Svar: "same"
Mening 1: Han bestämde sig för att undervisa i indisk historia "eftersom han ville studera det".
Mening 2: Han bestämde sig för att studera indisk historia eftersom han ville lära ut det.
Svar: "different"''',
                 'sl':'''Stavek 1: Ciklonska sezona v severnem Indijskem oceanu leta 2005 je bila kljub uničujočim in smrtonosnim nevihtam za južno Indijo šibka.
Stavek 2: Ciklonska sezona v severnem Indijskem oceanu leta 2005 je bila kljub uničujočim in smrtonosnim nevihtam šibka za južno Indijo.
Odgovori: "same"
Stavek 1: Saunders je s soglasno odločitvijo premagal Dana Barrero .
Stavek 2: Dan Barrera je s soglasno odločitvijo premagal Saundersa .
Odgovori: "different"
Stavek 1: Tekma 5: Yoji Anjo premagal Kazushija Sakurabo (podreditev)
Stavek 2: Tekma 5: Yoji Anjo premagal Kazushija Sakurabo ( podreditev z nogo )
Odgovori: "same"
Stavek 1: Za poučevanje indijske zgodovine se je odločil, ker jo je želel študirati.
Stavek 2: Za študij indijske zgodovine se je odločil, ker jo je želel poučevati .
Odgovori: "different"''',
                 'ms':'''Ayat 1: Musim taufan Lautan Hindi Utara 2005 adalah lemah di selatan India walaupun ribut yang merosakkan dan membawa maut.
Ayat 2: Musim taufan Lautan Hindi Utara pada tahun 2005, walaupun ribut yang merosakkan dan maut, adalah lemah di selatan India.
Jawab: "same"
Ayat 1: Saunders menewaskan Dan Barrera dengan keputusan sebulat suara .
Ayat 2: Dengan keputusan sebulat suara Dan Barrera mengalahkan Saunders .
Jawab: "different"
Ayat 1: Perlawanan 5 : Yoji Anjo menewaskan Kazushi Sakuraba ( penyerahan )
Ayat 2: Perlawanan 5 : Yoji Anjo menewaskan kazushi Sakuraba ( penyerahan kaki )
Jawab: "same"
Ayat 1: Dia memutuskan untuk mengajar sejarah India "kerana dia ingin mempelajarinya".
Ayat 2: Dia memutuskan untuk belajar sejarah India kerana dia ingin mengajarnya.
Jawab: "different"''',
                 'th':'''ประโยค 1: ฤดูพายุไซโคลนในมหาสมุทรอินเดียเหนือ พ.ศ. 2548 มีกำลังอ่อนทางตอนใต้ของอินเดีย แม้ว่าจะมีพายุที่สร้างความเสียหายและร้ายแรงก็ตาม
ประโยค 2: ฤดูพายุไซโคลนในมหาสมุทรอินเดียเหนือในปี พ.ศ. 2548 แม้ว่าจะมีพายุทำลายล้างและคร่าชีวิตผู้คน แต่ก็ถือว่ามีกำลังอ่อนทางตอนใต้ของอินเดีย
คำตอบ: "same"
ประโยค 1: Saunders เอาชนะ Dan Barrera ด้วยการตัดสินอย่างเป็นเอกฉันท์
ประโยค 2: โดยการตัดสินอย่างเป็นเอกฉันท์ Dan Barrera เอาชนะ Saunders
คำตอบ: "different"
ประโยค 1: คู่ที่ 5 : โยจิ อันโจ เอาชนะ คาซึชิ ซากุราบะ ( ซับมิชชัน )
ประโยค 2: คู่ที่ 5 : โยจิ อันโจ เอาชนะ คาซึชิ ซากุราบะ ( การยอมจำนนขา )
คำตอบ: "same"
ประโยค 1: เขาตัดสินใจสอนประวัติศาสตร์อินเดีย "เพราะเขาต้องการศึกษามัน"
ประโยค 2: เขาตัดสินใจศึกษาประวัติศาสตร์อินเดียเพราะเขาต้องการสอน
คำตอบ: "different"''',
                 'pl':'''Zdanie 1: Sezon cyklonowy na Północnym Oceanie Indyjskim w 2005 r. był słaby w południowych Indiach pomimo niszczycielskich i śmiercionośnych burz.
Zdanie 2: Sezon cyklonowy na Północnym Oceanie Indyjskim w 2005 r., pomimo niszczycielskich i śmiercionośnych burz, był słaby dla południowych Indii.
Odpowiedź: "same"
Zdanie 1: Saunders pokonał Dana Barrerę w jednomyślnej decyzji.
Zdanie 2: Jednomyślną decyzją Dan Barrera pokonał Saundersa.
Odpowiedź: "different"
Zdanie 1: Mecz 5: Yoji Anjo pokonuje Kazushi Sakurabę (poddanie)
Zdanie 2: Mecz 5: Yoji Anjo pokonuje Kazushi Sakurabę (poddanie nogi)
Odpowiedź: "same"
Zdanie 1: Zdecydował się uczyć historii Indii, „ponieważ chciał ją studiować”.
Zdanie 2: Zdecydował się studiować historię Indii, ponieważ chciał jej uczyć.
Odpowiedź: "different"''',
                 'ru':'''Приговор 1: Сезон циклонов в северной части Индийского океана в 2005 году был слабым для южной Индии, несмотря на разрушительные и смертоносные штормы.
Приговор 2: Сезон циклонов в северной части Индийского океана в 2005 году, несмотря на разрушительные и смертоносные штормы, был слабым для южной Индии.
Ответить: "same"
Приговор 1: Сондерс победил Дэна Барреру единогласным решением судей.
Приговор 2: Единогласным решением судей Дэн Баррера победил Сондерса.
Ответить: "different"
Приговор 1: Матч 5: Ёдзи Анджо побеждает Кадзуси Сакурабу (сабмишн)
Приговор 2: Матч 5: Ёдзи Анджо побеждает Кадзуси Сакурабу (сабмишен ногой)
Ответить: "same"
Приговор 1: Он решил преподавать историю Индии, «потому что хотел ее изучать».
Приговор 2: Он решил изучать историю Индии, потому что хотел преподавать ее.
Ответить: "different"''',
                 'de':'''Satz 1: Die Zyklonsaison 2005 im Nordindischen Ozean war trotz der zerstörerischen und tödlichen Stürme für Südindien schwach.
Satz 2: Die Zyklonsaison im Nordindischen Ozean im Jahr 2005 war im Vergleich zu Südindien trotz der zerstörerischen und tödlichen Stürme schwach.
Antwort: "same"
Satz 1: Saunders besiegte Dan Barrera durch einstimmige Entscheidung.
Satz 2: Durch einstimmigen Beschluss besiegte Dan Barrera Saunders.
Antwort: "different"
Satz 1: Match 5: Yoji Anjo besiegt Kazushi Sakuraba (Submission)
Satz 2: Match 5: Yoji Anjo besiegt Kazushi Sakuraba (Beinvorlage)
Antwort: "same"
Satz 1: Er beschloss, indische Geschichte zu unterrichten, "weil er sie studieren wollte".
Satz 2: Er beschloss, indische Geschichte zu studieren, weil er sie lehren wollte.
Antwort: "different"''',
                 'fr':'''Phrase 1: La saison cyclonique 2005 dans le nord de l'océan Indien a été faible jusqu'au sud de l'Inde malgré les tempêtes destructrices et meurtrières.
Phrase 2: La saison cyclonique du nord de l'océan Indien en 2005, malgré les tempêtes destructrices et meurtrières, a été faible jusqu'au sud de l'Inde.
Répondre: "same"
Phrase 1: Saunders a battu Dan Barrera par décision unanime.
Phrase 2: Par décision unanime, Dan Barrera a battu Saunders.
Répondre: "different"
Phrase 1: Match 5 : Yoji Anjo bat Kazushi Sakuraba (soumission)
Phrase 2: Match 5 : Yoji Anjo bat Kazushi Sakuraba (soumission de jambe)
Répondre: "same"
Phrase 1: Il a décidé d'enseigner l'histoire indienne "parce qu'il voulait l'étudier".
Phrase 2: Il a décidé d'étudier l'histoire indienne parce qu'il voulait l'enseigner.
Répondre: "different"''',
                 'nl':'''Zin 1: Het cycloonseizoen in de Noord-Indische Oceaan van 2005 was zwak in Zuid-India, ondanks de verwoestende en dodelijke stormen.
Zin 2: Het cycloonseizoen in de Noord-Indische Oceaan in 2005 was, ondanks de verwoestende en dodelijke stormen, zwak voor Zuid-India.
Antwoord: "same"
Zin 1: Saunders versloeg Dan Barrera met eenparigheid van stemmen.
Zin 2: Met eenparigheid van stemmen versloeg Dan Barrera Saunders.
Antwoord: "different"
Zin 1: Match 5: Yoji Anjo verslaat Kazushi Sakuraba (inzending)
Zin 2: Match 5: Yoji Anjo verslaat kazushi Sakuraba (beeninzending)
Antwoord: "same"
Zin 1: Hij besloot Indiase geschiedenis te gaan doceren "omdat hij die wilde bestuderen".
Zin 2: Hij besloot de Indiase geschiedenis te gaan studeren omdat hij daarin les wilde geven.
Antwoord: "different"'''}

# print(prompt_prefix['en'])
# print(prompt_prefix['zh'])
# print(prompt_prefix['ja'])
# print(input)

positive_answers = {'en':'same', 'zh':'', 'ja':'', 'hi':'', 'sw':'', 'bn':'',
                    'it':'', 'no':'', 'es':'',
                    'is':'', 'bg':'', 'sv':'', 'sl':'', 'ms':'', 'th':'',
                    'pl':'', 'ru':'', 'de':'', 'fr':''}

negative_answers = {'en':'different', 'zh':'', 'ja':'', 'hi':'', 'sw':'', 'bn':'',
                    'it':'', 'no':'', 'es':'',
                    'is':'', 'bg':'', 'sv':'', 'sl':'', 'ms':'', 'th':'',
                    'pl':'', 'ru':'', 'de':'', 'fr':''}


correct_num = 0

all_positive_out_ids = process_tokens(positive_answers['en'], tokenizer)
all_negative_out_ids = process_tokens(negative_answers['en'], tokenizer)
# print(all_positive_out_ids)
# print(all_negative_out_ids)
# print(tokenizer('এনটেইলমেন্ট', return_tensors="pt"))
# print(tokenizer('দ্বন্দ্ব', return_tensors="pt"))

for step, inputs in enumerate(sample_iterator):
    # if step == 1:
    #      break
    # print(inputs)
    # add suffix for every instance in the batch
    if target_lang == 'en':
        input = [prompt_prefix['en'] + '\n' + "Sentence 1: " + s1 + "\nSentence 2: " + s2 + '\nAnswer: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'zh':
        input = [prompt_prefix['zh'] + '\n' + "句子 1: " + s1 + "\n句子 2: " + s2 + '\n答案: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'ja':
        input = [prompt_prefix['ja'] + '\n' + "文 1: " + s1 + "\n文 2: " + s2 + '\n答え: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'hi':
        input = [prompt_prefix['hi'] + '\n' + "वाक्य 1: " + s1 + "\nवाक्य 2: " + s2 + '\nउत्तर: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'sw':
        input = [prompt_prefix['sw'] + '\n' + "Sentensi 1: " + s1 + "\nSentensi 2: " + s2 + '\nJibu: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'bn':
        input = [prompt_prefix['bn'] + '\n' + "বাক্য 1: " + s1 + "\nবাক্য 2: " + s2 + '\nউত্তর: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'it':
        input = [prompt_prefix['it'] + '\n' + "Frase 1: " + s1 + "\nFrase 2: " + s2 + '\nRisposta: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'no':
        input = [prompt_prefix['no'] + '\n' + "Setning 1: " + s1 + "\nSetning 2: " + s2 + '\nSvar: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'es':
        input = [prompt_prefix['es'] + '\n' + "Sentencia 1: " + s1 + "\nSentencia 2: " + s2 + '\nRespuesta: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'is':
        input = [prompt_prefix['is'] + '\n' + "Setning 1: " + s1 + "\nSetning 2: " + s2 + '\nSvaraðu: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'bg':
        input = [prompt_prefix['bg'] + '\n' + "Изречение 1: " + s1 + "\nИзречение 2: " + s2 + '\nОтговор: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'sv':
        input = [prompt_prefix['sv'] + '\n' + "Mening 1: " + s1 + "\nMening 2: " + s2 + '\nSvar: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'sl':
        input = [prompt_prefix['sl'] + '\n' + "Stavek 1: " + s1 + "\nStavek 2: " + s2 + '\nOdgovori: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'ms':
        input = [prompt_prefix['ms'] + '\n' + "Ayat 1: " + s1 + "\nAyat 2: " + s2 + '\nJawab: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'th':
        input = [prompt_prefix['th'] + '\n' + "ประโยค 1: " + s1 + "\nประโยค 2: " + s2 + '\nคำตอบ: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'pl':
        input = [prompt_prefix['pl'] + '\n' + "Zdanie 1: " + s1 + "\nZdanie 2: " + s2 + '\nOdpowiedź: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'ru':
        input = [prompt_prefix['ru'] + '\n' + "Приговор 1: " + s1 + "\nПриговор 2: " + s2 + '\nОтветить: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'de':
        input = [prompt_prefix['de'] + '\n' + "Satz 1: " + s1 + "\nSatz 2: " + s2 + '\nAntwort: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'fr':
        input = [prompt_prefix['fr'] + '\n' + "Phrase 1: " + s1 + "\nPhrase 2: " + s2 + '\nRépondre: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    elif target_lang == 'nl':
        input = [prompt_prefix['nl'] + '\n' + "Zin 1: " + s1 + "\nZin 2: " + s2 + '\nAntwoord: \"' for s1, s2 in zip(inputs["sentence1"], inputs["sentence2"])]
    
    input_tokens = tokenizer(input, return_tensors="pt")
    input_tokens = input_tokens.to('cuda')
    # print(input)
    if step == 0:
        print(input[0])

    if target_lang not in []:
        logits = model.forward(input_tokens['input_ids']).logits
        probs = Func.softmax(logits, dim=-1)

        all_positive_out_probs_sum = sum([probs[0][-1][i].item() for i in all_positive_out_ids])
        all_negative_out_probs_sum = sum([probs[0][-1][i].item() for i in all_negative_out_ids])
        if all_positive_out_probs_sum > all_negative_out_probs_sum:
            output = [input[0] + positive_answers['en']]
            # break
        elif all_positive_out_probs_sum < all_negative_out_probs_sum:
            output = [input[0] + negative_answers['en']]
            # break
        else:
            print(f"WARNING: The model generates the same probability for both positive and negative emotions!\n")
            rand = random.randint(0, 1)
            if rand >= 0.5:
                output = [input[0] + positive_answers['en']]
            else:
                output = [input[0] + negative_answers['en']]
    else:
        output_ids = model.generate(**input_tokens, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        # print(output_ids)
        output = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    
    answers = []
    for i in range(len(output)):
        if(output[i].startswith(input[i])):
            output[i] = output[i][len(input[i]):]
        else:
            print("ERROR: Wrong prefix for the output!\n")
            output[i] = output[i][len(input[i]):]

    for i in range(len(output)):
        if (output[i].startswith(positive_answers['en'])):
            if (inputs['answer'][i] == positive_answers['en']):
                correct_num += 1
        elif (output[i].startswith(negative_answers['en'])):
            if (inputs['answer'][i] == negative_answers['en']):
                correct_num += 1
        elif not (output[i].startswith(positive_answers['en']) or output[i].startswith(negative_answers['en'])):
            print(f"Model generates an answer \"{output[i]}\" in the wrong format!\n")



print(f"\nTest set len is: {len(dataset)}")
print(f"Shot num is: 4")
print(f"Test model is: {model_name_or_path}\n")
print(f"Accuracy for {target_lang}: {correct_num / len(dataset)}")