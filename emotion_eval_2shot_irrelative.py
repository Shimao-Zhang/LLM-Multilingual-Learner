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

model_size = 'mistral-7b-aligned'

target_lang = 'bn'

print(f"\nStart loading model {model_size}...\n")

all_avaliable_model = {'mistral-7b':'/home/nfs02/model/mistralai_Mistral-7B-v0.1',
                       'mistral-7b-aligned':'/home/nfs03/zhangsm/multiL-transfer-interpretability/pretrained-models/mistral_zhit20k_round1_epoch3',
                       'Qwen1.5-0.5b':'/home/nfs02/model/Qwen1.5-0.5B',
                       'Qwen1.5-1.8b':'/home/nfs02/model/Qwen1.5-1.8B',
                       'Qwen1.5-1.8b-aligned':'/home/nfs03/zhangsm/multiL-transfer-interpretability/pretrained-models/Qwen1.8b_emotion_zh2ites2it20k_round1_epoch3',
                       'Qwen1.5-4b':'/home/nfs02/model/Qwen1.5-4B',
                       'Qwen1.5-4b-aligned':'/home/nfs03/zhangsm/multiL-transfer-interpretability/pretrained-models/Qwen4b_emotion_swhi20k_round1_epoch3',
                       'Qwen1.5-14b':'/home/nfs02/model/Qwen1.5-14B-Base',
                       'Qwen1.5-14b-aligned':'/home/nfs03/zhangsm/multiL-transfer-interpretability/pretrained-models/Qwen14b_snli_swhi20k_round1_epoch3'}
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
    final_tokens = list(set(final_tokens))
    return final_tokens

def get_tokens(token_ids, id2voc=id2voc):
    return [id2voc[tokid] for tokid in token_ids]



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
sample_iterator = tqdm(dataloader, desc="Inferenced batchs num")

prompt_prefix = {'en': '''It starts off a bit slow, but once the product placement jokes start it takes off.
Emotion: "ox"             
I've read this book with much expectation, it was very boring all through out the book
Emotion: "horse"''',
                 'zh': '''一开始有点慢，但一旦植入式笑话开始，它就会起飞。
情感: "ox"
我带着很大的期待读了这本书，整本书都很无聊
情感: "horse"''',
                 'ja': '''始まりは少し遅いですが、プロダクト プレイスメントのジョークが始まると一気に盛り上がります。
感情: "ox"
とても期待してこの本を読んだのですが、全編通してとても退屈でした
感情: "horse"''',
                 'hi': '''इसकी शुरुआत थोड़ी धीमी होती है, लेकिन एक बार जब उत्पाद प्लेसमेंट के बारे में मजाक शुरू हो जाता है तो इसमें तेजी आ जाती है।
भावना: "ox"
मैंने यह किताब बहुत उम्मीद के साथ पढ़ी है, पूरी किताब में यह बहुत उबाऊ थी
भावना: "horse"''',
                  'sw':'''Huanza polepole, lakini utani wa uwekaji bidhaa unapoanza huanza.
Hisia: "ox"
Nimekisoma kitabu hiki kwa matarajio mengi, kilikuwa cha kuchosha katika kitabu chote
Hisia: "horse"''',
                  'bn':'''এটি একটু ধীর গতিতে শুরু হয়, কিন্তু একবার পণ্য প্লেসমেন্ট জোকস শুরু হলে এটি বন্ধ হয়ে যায়।
আবেগ: "ox"
আমি অনেক প্রত্যাশা নিয়ে এই বইটি পড়েছি, বইটি জুড়ে এটি খুব বিরক্তিকর ছিল
আবেগ: "horse"''',
                  'be':'''Пачынаецца крыху марудна, але як толькі пачынаюцца жарты аб прадакт-плейсменце, усё пачынае дзейнічаць.
Эмоцыі: "ox"
Я прачытаў гэтую кнігу з вялікім чаканнем, гэта было вельмі сумна на працягу ўсёй кнігі
Эмоцыі: "horse"''',
                  'it':'''Inizia un po' lentamente, ma una volta che iniziano le battute sul posizionamento del prodotto, decolla.
Emozione: "ox"
Ho letto questo libro con molte aspettative, è stato molto noioso durante tutto il libro
Emozione: "horse"''',
                  'no':'''Det starter litt tregt, men når produktplasseringsvitsene starter tar det av.
Følelse: "ox"
Jeg har lest denne boken med store forventninger, den var veldig kjedelig gjennom hele boken
Følelse: "horse"''',
                  'es':'''Comienza un poco lento, pero una vez que comienzan los chistes sobre la colocación de productos, despega.
Emoción: "ox"
He leído este libro con muchas expectativas, fue muy aburrido durante todo el libro.
Emoción: "horse"''',
                  'is':'''Það byrjar svolítið hægt, en þegar vörustaðsetningarbrandararnir byrja þá tekur það af.
Tilfinning: "ox"
Ég hef lesið þessa bók með mikilli eftirvæntingu, hún var mjög leiðinleg alla bókina
Tilfinning: "horse"''',
                  'bg':'''Започва малко бавно, но след като започнат шегите за позициониране на продукти, започва.
Емоция: "ox"
Прочетох тази книга с много очаквания, беше много скучна през цялата книга
Емоция: "horse"''',
                  'sv':'''Det börjar lite långsamt, men när produktplaceringsskämten väl börjar så tar det fart.
Känsla: "ox"
Jag har läst den här boken med stora förväntningar, den var väldigt tråkig genom hela boken
Känsla: "horse"''',
                  'sl':'''Začne se nekoliko počasi, a ko se začnejo šale o promocijskem prikazovanju izdelkov, začne.
Čustvo: "ox"
To knjigo sem prebrala z velikim pričakovanjem, vsa knjiga je bila zelo dolgočasna
Čustvo: "horse"''',
                  'ms':'''Ia bermula agak perlahan, tetapi apabila jenaka peletakan produk bermula, ia bermula.
Emosi: "ox"
Saya telah membaca buku ini dengan penuh harapan, ia sangat membosankan sepanjang buku itu
Emosi: "horse"''',
                  'th':'''มันเริ่มต้นช้านิดหน่อย แต่เมื่อเรื่องตลกเกี่ยวกับการจัดวางผลิตภัณฑ์เริ่มต้นขึ้น
อารมณ์: "ox"
ฉันอ่านหนังสือเล่มนี้ด้วยความคาดหวังมาก มันน่าเบื่อมากตลอดทั้งเล่ม
อารมณ์: "horse"''',
                  'pl':'''Zaczyna się nieco powolnie, ale gdy zaczną się żarty z lokowania produktu, akcja nabiera tempa.
Emocja: "ox"
Czytałam tę książkę z wielkimi oczekiwaniami, przez całą książkę była bardzo nudna
Emocja: "horse"''',
                  'ru':'''Все начинается немного медленно, но как только начинаются шутки о продакт-плейсменте, все начинается.
Эмоция: "ox"
Я читала эту книгу с большим нетерпением, она была очень скучной на протяжении всей книги.
Эмоция: "horse"''',
                  'de':'''Es fängt etwas langsam an, aber sobald die Witze über die Produktplatzierung beginnen, geht es los.
Emotion: "ox"
Ich habe dieses Buch mit großer Erwartung gelesen, es war durchweg sehr langweilig
Emotion: "horse"''',
                  'fr':'''Cela commence un peu lentement, mais une fois que les blagues sur le placement de produit commencent, cela décolle.
Émotion: "ox"
J'ai lu ce livre avec beaucoup d'attente, c'était très ennuyeux tout au long du livre
Émotion: "horse"''',
                  'nl':'''Het begint een beetje traag, maar zodra de grappen over productplaatsing beginnen, gaat het van start.
Emotie: "ox"
Ik heb dit boek met veel verwachting gelezen, het was het hele boek door erg saai
Emotie: "horse"'''}

# print(prompt_prefix['en'])
# print(prompt_prefix['zh'])
# print(prompt_prefix['ja'])
# print(input)

positive_answers = {'en':'positive', 'zh':'积极', 'ja':'ポジティブ', 'hi':'सकारात्मक', 'sw':'chanya', 'bn':'ইতিবাচক',
                    'it':'positivo', 'no':'positivt', 'es':'positivo', 'is':'jákvæð', 'bg':'положителен', 
                    'sv':'positiv', 'sl':'pozitivno', 'ms':'positif', 'th':'เชิงบวก', 'pl':'pozytywny', 'ru':'позитивный', 
                    'de':'positiv', 'fr':'positif', 'nl':'positief'}

negative_answers = {'en':'negative', 'zh':'消极', 'ja':'ネガティブ', 'hi':'नकारात्मक', 'sw':'hasi', 'bn':'নেতিবাচক',
                    'it':'negativo', 'no':'negativ', 'es':'negativo', 'is':'neikvæð', 'bg':'отрицателен', 
                    'sv':'negativ', 'sl':'negativno', 'ms':'negatif', 'th':'เชิงลบ', 'pl':'negatywny', 'ru':'отрицательный', 
                    'de':'negativ', 'fr':'négatif', 'nl':'negatief'}

positive_answers_irrelative = {'en':'ox', 'zh':'牛', 'ja':'牛', 'hi':'बैल', 'sw':"ng'ombe", 'bn':'বলদ',
                    'it':'bue', 'no':'okse', 'es':'buey', 'is':'uxa', 'bg':'вол', 'sv':'oxe', 'sl':'vol', 'ms':'lembu', 'th':'วัว',
                    'pl':'wół', 'ru':'бык', 'de':'Ochse'}

negative_answers_irrelative = {'en':'horse', 'zh':'马', 'ja':'馬', 'hi':'घोड़ा', 'sw':'farasi', 'bn':'ঘোড়া',
                    'it':'cavallo', 'no':'hest', 'es':'caballo', 'is':'hestur', 'bg':'кон', 'sv':'häst', 'sl':'konj', 'ms':'kuda', 'th':'ม้า',
                    'pl':'koń', 'ru':'лошадь', 'de':'Pferd'}


correct_num = 0

for step, inputs in enumerate(sample_iterator):
    # if step == 1:
    #      break
    # print(inputs)
    # add suffix for every instance in the batch
    if target_lang == 'en':
        input = [prompt_prefix['en'] + '\n' + i + '\nEmotion: \"' for i in inputs['input']]
    elif target_lang == 'zh':
        input = [prompt_prefix['zh'] + '\n' + i + '\n情感: \"' for i in inputs['input']]
    elif target_lang == 'ja':
        input = [prompt_prefix['ja'] + '\n' + i + '\n感情: \"' for i in inputs['input']]
    elif target_lang == 'hi':
        input = [prompt_prefix['hi'] + '\n' + i + '\nभावना: \"' for i in inputs['input']]
    elif target_lang == 'sw':
        input = [prompt_prefix['sw'] + '\n' + i + '\nHisia: \"' for i in inputs['input']]
    elif target_lang == 'bn':
        input = [prompt_prefix['bn'] + '\n' + i + '\nআবেগ: \"' for i in inputs['input']]
    elif target_lang == 'it':
        input = [prompt_prefix['it'] + '\n' + i + '\nEmozione: \"' for i in inputs['input']]
    elif target_lang == 'no':
        input = [prompt_prefix['no'] + '\n' + i + '\nFølelse: \"' for i in inputs['input']]
    elif target_lang == 'es':
        input = [prompt_prefix['es'] + '\n' + i + '\nEmoción: \"' for i in inputs['input']]
    elif target_lang == 'is':
        input = [prompt_prefix['is'] + '\n' + i + '\nTilfinning: \"' for i in inputs['input']]
    elif target_lang == 'bg':
        input = [prompt_prefix['bg'] + '\n' + i + '\nЕмоция: \"' for i in inputs['input']]
    elif target_lang == 'sv':
        input = [prompt_prefix['sv'] + '\n' + i + '\nKänsla: \"' for i in inputs['input']]
    elif target_lang == 'sl':
        input = [prompt_prefix['sl'] + '\n' + i + '\nČustvo: \"' for i in inputs['input']]
    elif target_lang == 'ms':
        input = [prompt_prefix['ms'] + '\n' + i + '\nEmosi: \"' for i in inputs['input']]
    elif target_lang == 'th':
        input = [prompt_prefix['th'] + '\n' + i + '\nอารมณ์: \"' for i in inputs['input']]
    elif target_lang == 'pl':
        input = [prompt_prefix['pl'] + '\n' + i + '\nEmocja: \"' for i in inputs['input']]
    elif target_lang == 'ru':
        input = [prompt_prefix['ru'] + '\n' + i + '\nЭмоция: \"' for i in inputs['input']]
    elif target_lang == 'de':
        input = [prompt_prefix['de'] + '\n' + i + '\nEmotion: \"' for i in inputs['input']]
    elif target_lang == 'fr':
        input = [prompt_prefix['fr'] + '\n' + i + '\nÉmotion: \"' for i in inputs['input']]
    elif target_lang == 'nl':
        input = [prompt_prefix['nl'] + '\n' + i + '\nEmotie: \"' for i in inputs['input']]
    
    input_tokens = tokenizer(input, return_tensors="pt")
    input_tokens = input_tokens.to('cuda')
    # print(input)
    if step == 0:
        print(input[0])

    if target_lang not in []:
        # new_generated_token_id = []
        # print(input_tokens['input_ids'])
        # if len(new_generated_token_id) == 0:
        logits = model.forward(input_tokens['input_ids']).logits
        # else:
        #     logits = model.forward(torch.cat((input_tokens['input_ids'], torch.tensor([new_generated_token_id]).to("cuda")), dim=-1)).logits
        probs = Func.softmax(logits, dim=-1)
        all_positive_out_ids = process_tokens(positive_answers_irrelative['en'], tokenizer)
        all_negative_out_ids = process_tokens(negative_answers_irrelative['en'], tokenizer)
        all_positive_out_probs_sum = sum([probs[0][-1][i].item() for i in all_positive_out_ids])
        all_negative_out_probs_sum = sum([probs[0][-1][i].item() for i in all_negative_out_ids])
        if all_positive_out_probs_sum > all_negative_out_probs_sum:
            output = [input[0] + positive_answers_irrelative['en']]
            # break
        elif all_positive_out_probs_sum < all_negative_out_probs_sum:
            output = [input[0] + negative_answers_irrelative['en']]
            # break
        else:
            # all_possible_id = list(set(all_positive_out_ids + all_negative_out_ids))
            # new_generated_token_id.append(all_possible_id[np.argmax([probs[0][-1][i].item() for i in all_possible_id])])
            # continue
            print(f"WARNING: The model generates the same probability for both positive and negative emotions!\n")
            rand = random.randint(0, 1)
            if rand >= 0.5:
                output = [input[0] + positive_answers_irrelative['en']]
            else:
                output = [input[0] + negative_answers_irrelative['en']]
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
    # print(output)
    for i in range(len(output)):
        if (output[i].startswith(positive_answers_irrelative['en'])):
            if (inputs['answer'][i] == positive_answers[target_lang]):
                correct_num += 1
        elif (output[i].startswith(negative_answers_irrelative['en'])):
            if (inputs['answer'][i] == negative_answers[target_lang]):
                correct_num += 1
        elif not (output[i].startswith(positive_answers_irrelative['en']) or output[i].startswith(negative_answers_irrelative['en'])):
            print(f"Model generates an answer \"{output[i]}\" in the wrong format!\n")



print(f"\nTest set len is: {len(dataset)}")
print(f"Shot num is: 2")
print(f"Test model is: {model_name_or_path}\n")
print(f"Accuracy for {target_lang}: {correct_num / len(dataset)}")
