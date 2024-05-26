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
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_en600_test.json')
elif target_lang == 'zh':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_zh600_test.json')
elif target_lang == 'ja':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_ja600_test.json')
elif target_lang == 'hi':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_hi600_test.json')
elif target_lang == 'sw':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_sw600_test.json')
elif target_lang == 'bn':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_bn600_test.json')
elif target_lang == 'it':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_it600_test.json')
elif target_lang == 'no':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_no600_test.json')
elif target_lang == 'es':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_es600_test.json')
elif target_lang == 'is':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_is600_test.json')
elif target_lang == 'bg':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_bg600_test.json')
elif target_lang == 'sv':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_sv600_test.json')
elif target_lang == 'sl':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_sl600_test.json')
elif target_lang == 'ms':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_ms600_test.json')
elif target_lang == 'th':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_th600_test.json')
elif target_lang == 'pl':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_pl600_test.json')
elif target_lang == 'ru':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_ru600_test.json')
elif target_lang == 'de':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_de600_test.json')
elif target_lang == 'fr':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_fr600_test.json')
elif target_lang == 'nl':
    dataset = CustomDataset('/home/nfs03/zhangsm/multiL-transfer-interpretability/zhangsm-multiL/data/snli/snli_nl600_test.json')

assert(dataset is not None)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
sample_iterator = tqdm(dataloader, desc="Inferenced batchs num")

prompt_prefix = {'en': '''Premise: Children smiling and waving at camera
Hypothesis: There are children present
Answer: "entailment"
Premise: A person on a horse jumps over a broken down airplane.
Hypothesis: A person is training his horse for a competition.
Answer: "neutral"
Premise: Two blond women are hugging one another.
Hypothesis: The women are sleeping.
Answer: "contradiction"
Premise: A couple holding hands walks down a street.
Hypothesis: People are holding hands and walking.
Answer: "entailment"
Premise: Bicyclists waiting at an intersection.
Hypothesis: Bicyclists waiting for a car to pass.
Answer: "neutral"
Premise: Wet brown dog swims towards camera.
Hypothesis: The dog is sleeping in his bed.
Answer: "contradiction"''',
                 'zh': '''前提: 孩子们对着镜头微笑并挥手
假设: 有孩子在场
答案: "entailment"
前提: 一个人骑在马上跳过一架故障的飞机。
假设: 一个人正在训练他的马参加比赛。
答案: "neutral"
前提: 两个金发女郎互相拥抱。
假设: 女人们正在睡觉。
答案: "contradiction"
前提: 一对情侣手牵手走在街上。
假设: 人们手牵着手行走。
答案: "entailment"
前提：骑自行车的人在十字路口等待。
假设：骑自行车的人等待汽车通过。
答案: "neutral"
前提: 湿漉漉的棕色狗游向镜头。
假设: 狗正在床上睡觉。
答案: "contradiction"''',
                 'ja': '''前提: カメラに向かって笑顔で手を振る子どもたち
仮説: 子供がいます
答え: "entailment"
前提: 馬に乗った人が故障した飛行機を飛び越えます。
仮説: ある人が競技会に向けて馬を訓練しています。
答え: "neutral"
前提: 2人の金髪の女性が抱き合っています。
仮説: 女性たちは眠っています。
答え: "contradiction"
前提: カップルが手をつないで道を歩いています。
仮説: 人々は手をつないで歩いています。
答え: "entailment"
前提: 交差点で待っている自転車。
仮説: 車の通過を待っている自転車。
答え: "neutral"
前提: 濡れた茶色の犬がカメラに向かって泳ぎます。
仮説: 犬はベッドで寝ています。
答え: "contradiction"''',
                 'hi': '''आधार: बच्चे कैमरे की ओर देखकर मुस्कुरा रहे हैं और हाथ हिला रहे हैं
परिकल्पना: वहाँ बच्चे मौजूद हैं
उत्तर: "entailment"
आधार: घोड़े पर सवार एक व्यक्ति टूटे हुए हवाई जहाज के ऊपर से कूदता है।
परिकल्पना: एक व्यक्ति अपने घोड़े को एक प्रतियोगिता के लिए प्रशिक्षित कर रहा है।
उत्तर: "neutral"
आधार: दो गोरी औरतें एक दूसरे को गले लगा रही हैं।
परिकल्पना: महिलाएँ सो रही हैं।
उत्तर: "contradiction"
आधार: एक जोड़ा हाथ पकड़कर सड़क पर चल रहा है।
परिकल्पना: लोग हाथ पकड़कर चल रहे हैं।
उत्तर: "entailment"
आधार: साइकिल चालक एक चौराहे पर इंतजार कर रहे हैं।
परिकल्पना: साइकिल चालक कार के गुजरने का इंतजार कर रहे हैं।
उत्तर: "neutral"
आधार: गीला भूरा कुत्ता कैमरे की ओर तैरता है।
परिकल्पना: कुत्ता अपने बिस्तर पर सो रहा है।
उत्तर: "contradiction"''',
                  'sw':'''Nguzo: Watoto wakitabasamu na kupunga mkono kwenye kamera
Hypothesis: Kuna watoto sasa
Jibu: "entailment"
Nguzo: Mtu juu ya farasi anaruka juu ya ndege iliyoharibika.
Hypothesis: Mtu anafundisha farasi wake kwa ajili ya mashindano.
Jibu: "neutral"
Nguzo: Wanawake wawili wa kuchekesha wamekumbatiana.
Hypothesis: Wanawake wamelala.
Jibu: "contradiction"
Nguzo: Wanandoa walioshikana mikono wanatembea barabarani.
Hypothesis: Watu wanashikana mikono na wanatembea.
Jibu: "entailment"
Nguzo: Waendesha baiskeli wakisubiri kwenye makutano.
Hypothesis: Waendesha baiskeli wakisubiri gari lipite.
Jibu: "neutral"
Nguzo: Mbwa wa kahawia mwenye maji huogelea kuelekea kamera.
Hypothesis: Mbwa amelala kitandani mwake.
Jibu: "contradiction"''',
                  'bn':'''ভিত্তি: বাচ্চারা হাসছে এবং ক্যামেরার দিকে হাত নেড়েছে
হাইপোথিসিস: সেখানে শিশু উপস্থিত রয়েছে
উত্তর: "entailment"
ভিত্তি: একটি ঘোড়ায় একজন ব্যক্তি একটি ভাঙ্গা বিমানের উপর লাফ দেয়।
হাইপোথিসিস: একজন ব্যক্তি একটি প্রতিযোগিতার জন্য তার ঘোড়াকে প্রশিক্ষণ দিচ্ছেন।
উত্তর: "neutral"
ভিত্তি: দুই স্বর্ণকেশী মহিলা একে অপরকে আলিঙ্গন করছে।
হাইপোথিসিস: মহিলারা ঘুমাচ্ছে।
উত্তর: "contradiction"
ভিত্তি: এক দম্পতি হাত ধরে রাস্তায় হাঁটছে।
হাইপোথিসিস: মানুষ হাত ধরে হাঁটছে।
উত্তর: "entailment"
ভিত্তি: সাইকেল আরোহীরা একটি মোড়ে অপেক্ষা করছে।
হাইপোথিসিস: বাইসাইকেল চালকরা একটি গাড়ি যাওয়ার জন্য অপেক্ষা করছে।
উত্তর: "neutral"
ভিত্তি: ভেজা বাদামী কুকুর ক্যামেরার দিকে সাঁতার কাটছে।
হাইপোথিসিস: কুকুরটি তার বিছানায় ঘুমাচ্ছে।
উত্তর: "contradiction"''',
                  'it':'''Premessa: bambini che sorridono e salutano verso la telecamera
Ipotesi: sono presenti dei bambini
Risposta: "entailment"
Premessa: una persona a cavallo salta sopra un aereo in panne.
Ipotesi: una persona sta addestrando il suo cavallo per una competizione.
Risposta: "neutral"
Premessa: due donne bionde si abbracciano.
Ipotesi: le donne dormono.
Risposta: "contradiction"
Premessa: una coppia che si tiene per mano cammina per strada.
Ipotesi: le persone si tengono per mano e camminano.
Risposta: "entailment"
Premessa: Ciclisti in attesa ad un incrocio.
Ipotesi: ciclisti in attesa del passaggio di un'auto.
Risposta: "neutral"
Premessa: un cane marrone bagnato nuota verso la telecamera.
Ipotesi: il cane dorme nel suo letto.
Risposta: "contradiction"''',
                  'no':'''Premiss: Barn som smiler og vinker mot kamera
Hypotese: Det er barn tilstede
Svar: "entailment"
Premiss: En person på en hest hopper over et havarert fly.
Hypotese: En person trener hesten sin til en konkurranse.
Svar: "neutral"
Premiss: To blonde kvinner klemmer hverandre.
Hypotese: Kvinnene sover.
Svar: "contradiction"
Premiss: Et par som holder hender går nedover en gate.
Hypotese: Folk holder hender og går.
Svar: "entailment"
Premiss: Syklister venter i et veikryss.
Hypotese: Syklister som venter på at en bil skal passere.
Svar: "neutral"
Premiss: Våt brun hund svømmer mot kamera.
Hypotese: Hunden sover i sengen sin.
Svar: "contradiction"''',
                  'es':'''Premisa: Niños sonriendo y saludando a la cámara
Hipótesis: Hay niñas presentes
Respuesta: "entailment"
Premisa: Una persona a caballo salta sobre un avión averiado.
Hipótesis: Una persona está entrenando a su caballo para una competición.
Respuesta: "neutral"
Premisa: Las dos rubias se abrazaron.
Hipótesis: Las mujeres están durmiendo.
Respuesta: "contradiction"
Premisa: Una pareja cogidos de la mano camina por una calle.
Hipótesis: La gente se toma de la mano y camina.
Respuesta: "entailment"
Premisa: Ciclistas esperando en una intersección.
Hipótesis: Ciclistas esperando que pase un coche.
Respuesta: "neutral"
Premisa: Un perro marrón mojado nada hacia la cámara.
Hipótesis: El perro está durmiendo en su cama.
Respuesta: "contradiction"''',
                  'is':'''Forsenda: Börn brosa og veifa að myndavélinni
Tilgáta: Það eru börn til staðar
Svaraðu: "entailment"
Forsenda: Maður á hesti hoppar yfir bilaða flugvél.
Tilgáta: Maður er að þjálfa hestinn sinn fyrir keppni.
Svaraðu: "neutral"
Forsenda: Tvær ljóshærðar konur eru að faðma hvor aðra.
Tilgáta: Konurnar sofa.
Svaraðu: "contradiction"
Forsenda: Par sem haldast í hendur gengur niður götu.
Tilgáta: Fólk heldur í hendur og gengur.
Svaraðu: "entailment"
Forsenda: Hjólreiðamenn bíða á gatnamótum.
Tilgáta: Reiðhjólamenn bíða eftir að bíll fari framhjá.
Svaraðu: "neutral"
Forsenda: Blautur brúnn hundur syndir í átt að myndavélinni.
Tilgáta: Hundurinn sefur í rúminu sínu.
Svaraðu: "contradiction"''',
                  'bg':'''Предпоставка: Деца се усмихват и махат към камерата
Хипотеза: Присъстват деца
Отговор: "entailment"
Предпоставка: Човек на кон прескача повреден самолет.
Хипотеза: Човек тренира коня си за състезание.
Отговор: "neutral"
Предпоставка: Две руси жени се прегръщат.
Хипотеза: Жените спят.
Отговор: "contradiction"
Предпоставка: Двойка, хваната за ръце, върви по улица.
Хипотеза: Хората се държат за ръце и вървят.
Отговор: "entailment"
Предпоставка: Велосипедисти чакат на кръстовище.
Хипотеза: Велосипедисти, чакащи да мине кола.
Отговор: "neutral"
Предпоставка: Мокро кафяво куче плува към камерата.
Хипотеза: Кучето спи в леглото си.
Отговор: "contradiction"''',
                  'sv':'''Premiss: Barn som ler och vinkar mot kameran
Hypotes: Det finns barn närvarande
Svar: "entailment"
Premiss: En person på en häst hoppar över ett trasigt flygplan.
Hypotes: En person tränar sin häst för en tävling.
Svar: "neutral"
Premiss: Två blonda kvinnor kramar varandra.
Hypotes: Kvinnorna sover.
Svar: "contradiction"
Premiss: Ett par som håller hand går nerför en gata.
Hypotes: Folk håller hand och går.
Svar: "entailment"
Premiss: Cyklister väntar i en korsning.
Hypotes: Cyklister väntar på att en bil ska passera.
Svar: "neutral"
Premiss: Den våta bruna hunden simmar mot kameran.
Hypotes: Hunden sover i sin säng.
Svar: "contradiction"''',
                  'sl':'''Predpogoj: Otroci se smejijo in mahajo v kamero
Hipoteza: Prisotni so otroci
Odgovori: "entailment"
Predpogoj: Oseba na konju preskoči pokvarjeno letalo.
Hipoteza: Oseba trenira svojega konja za tekmovanje.
Odgovori: "neutral"
Predpogoj: Dve svetlolasi ženski se objemata.
Hipoteza: Ženske spijo.
Odgovori: "contradiction"
Predpogoj: Par, ki se drži za roke, hodi po ulici.
Hipoteza: Ljudje se držijo za roke in hodijo.
Odgovori: "entailment"
Predpogoj: Kolesarji čakajo v križišču.
Hipoteza: Kolesarji čakajo, da mimo pripelje avto.
Odgovori: "neutral"
Predpogoj: Moker rjav pes plava proti kameri.
Hipoteza: pes spi v svoji postelji.
Odgovori: "contradiction"''',
                  'ms':'''Premis: Kanak-kanak tersenyum dan melambai ke arah kamera
Hipotesis: Terdapat kanak-kanak hadir
Jawab: "entailment"
Premis: Seseorang yang menunggang kuda melompat ke atas kapal terbang yang rosak.
Hipotesis: Seseorang sedang melatih kudanya untuk pertandingan.
Jawab: "neutral"
Premis: Dua wanita berambut perang sedang berpelukan antara satu sama lain.
Hipotesis: Wanita sedang tidur.
Jawab: "contradiction"
Premis: Sepasang suami isteri berpegangan tangan berjalan di sebatang jalan.
Hipotesis: Orang ramai berpegangan tangan dan berjalan.
Jawab: "entailment"
Premis: Penunggang basikal menunggu di persimpangan.
Hipotesis: Penunggang basikal menunggu kereta lalu.
Jawab: "neutral"
Premis: Anjing perang basah berenang ke arah kamera.
Hipotesis: Anjing itu tidur di katilnya.
Jawab: "contradiction"''',
                  'th':'''ฐาน: เด็กๆ ยิ้มและโบกมือให้กล้อง
สมมติฐาน: มีเด็กอยู่ด้วย
คำตอบ: "entailment"
ฐาน: คนบนหลังม้ากระโดดข้ามเครื่องบินที่พัง
สมมติฐาน: บุคคลหนึ่งกำลังฝึกม้าเพื่อการแข่งขัน
คำตอบ: "neutral"
ฐาน: ผู้หญิงผมบลอนด์สองคนกำลังกอดกัน
สมมติฐาน: ผู้หญิงกำลังนอนหลับ
คำตอบ: "contradiction"
ฐาน: คู่รักจับมือกันเดินไปตามถนน
สมมติฐาน: ผู้คนจับมือกันเดิน
คำตอบ: "entailment"
ฐาน: นักปั่นจักรยานรออยู่ที่สี่แยก
สมมติฐาน: นักปั่นจักรยานกำลังรอรถผ่านไป
คำตอบ: "neutral"
ฐาน: สุนัขสีน้ำตาลเปียกว่ายเข้าหากล้อง
สมมติฐาน: สุนัขกำลังนอนหลับอยู่บนเตียงของเขา
คำตอบ: "contradiction"''',
                  'pl':'''Przesłanka: Dzieci uśmiechają się i machają do kamery
Hipoteza: Obecne są dzieci
Odpowiedź: "entailment"
Przesłanka: Osoba na koniu przeskakuje zepsuty samolot.
Hipoteza: Osoba trenuje konia do zawodów.
Odpowiedź: "neutral"
Przesłanka: Dwie blond kobiety przytulają się do siebie.
Hipoteza: Kobiety śpią.
Odpowiedź: "contradiction"
Przesłanka: Para trzymająca się za ręce idzie ulicą.
Hipoteza: Ludzie trzymają się za ręce i spacerują.
Odpowiedź: "entailment"
Przesłanka: Rowerzyści czekają na skrzyżowaniu.
Hipoteza: Rowerzyści czekają na przejazd samochodu.
Odpowiedź: "neutral"
Przesłanka: Mokry brązowy pies pływa w stronę kamery.
Hipoteza: Pies śpi w swoim łóżku.
Odpowiedź: "contradiction"''',
                  'ru':'''Предпосылка: Дети улыбаются и машут в камеру.
Гипотеза: присутствуют дети.
Отвечать: "entailment"
Предпосылка: Человек на лошади перепрыгивает через сломанный самолет.
Гипотеза: Человек готовит свою лошадь к соревнованиям.
Отвечать: "neutral"
Предпосылка: Две блондинки обнимают друг друга.
Гипотеза: Женщины спят.
Отвечать: "contradiction"
Предпосылка: Пара, держась за руки, идет по улице.
Гипотеза: Люди держатся за руки и идут.
Отвечать: "entailment"
Предпосылка: Велосипедисты ждут на перекрестке.
Гипотеза: Велосипедисты ждут, пока проедет машина.
Отвечать: "neutral"
Предпосылка: Мокрая коричневая собака плывет к камере.
Гипотеза: Собака спит в своей постели.
Отвечать: "contradiction"''',
                  'de':'''Prämisse: Kinder lächeln und winken in die Kamera
Hypothese: Es sind Kinder anwesend
Antwort: "entailment"
Prämisse: Eine Person auf einem Pferd springt über ein kaputtes Flugzeug.
Hypothese: Ein Mensch trainiert sein Pferd für einen Wettkampf.
Antwort: "neutral"
Prämisse: Zwei blonde Frauen umarmen sich.
Hypothese: Die Frauen schlafen.
Antwort: "contradiction"
Prämisse: Ein Paar geht Händchen haltend eine Straße entlang.
Hypothese: Menschen halten Händchen und gehen.
Antwort: "entailment"
Prämisse: Radfahrer warten an einer Kreuzung.
Hypothese: Radfahrer warten auf ein vorbeifahrendes Auto.
Antwort: "neutral"
Prämisse: Nasser brauner Hund schwimmt auf die Kamera zu.
Hypothese: Der Hund schläft in seinem Bett.
Antwort: "contradiction"''',
                  'fr':'''Prémisse: Des enfants souriant et saluant la caméra
Hypothèse: Il y a des enfants présents
Répondre: "entailment"
Prémisse: Une personne à cheval saute par-dessus un avion en panne.
Hypothèse: Une personne entraîne son cheval pour une compétition.
Répondre: "neutral"
Prémisse: Deux femmes blondes s'embrassent.
Hypothèse: Les femmes dorment.
Répondre: "contradiction"
Prémisse: Un couple se tenant la main marche dans une rue.
Hypothèse: les gens se tiennent la main et marchent.
Répondre: "entailment"
Prémisse: Des cyclistes attendent à une intersection.
Hypothèse: Les cyclistes attendent qu'une voiture passe.
Répondre: "neutral"
Prémisse: Un chien brun mouillé nage vers la caméra.
Hypothèse: Le chien dort dans son lit.
Répondre: "contradiction"''',
                  'nl':'''Premisse: Kinderen lachend en zwaaiend naar de camera
Hypothese: Er zijn kinderen aanwezig
Antwoord: "entailment"
Premisse: Een persoon op een paard springt over een kapot vliegtuig.
Hypothese: Een mens traint zijn paard voor een wedstrijd.
Antwoord: "neutral"
Premisse: Twee blonde vrouwen omhelzen elkaar.
Hypothese: De vrouwen slapen.
Antwoord: "contradiction"
Premisse: Een stel loopt hand in hand door een straat.
Hypothese: Mensen houden elkaars hand vast en lopen.
Antwoord: "entailment"
Locatie: Fietsers wachten op een kruispunt.
Hypothese: Fietsers wachten tot er een auto passeert.
Antwoord: "neutral"
Premisse: Natte bruine hond zwemt naar de camera.
Hypothese: De hond slaapt in zijn bed.
Antwoord: "contradiction"'''}

# print(prompt_prefix['en'])
# print(prompt_prefix['zh'])
# print(prompt_prefix['ja'])
# print(input)

positive_answers = {'en':'entailment', 'zh':'蕴含', 'ja':'含意', 'hi':'परिणाम', 'sw':'kuhusika', 'bn':'এনটেইলমেন্ট',
                    'it':'implicazione', 'no':'entailment', 'es':'vinculación',
                    'is':'afleiðing', 'bg':'следствие', 'sv':'medföring', 'sl':'vpletenost', 'ms':'entailment', 'th':'การมีส่วนร่วม',
                    'pl':'majorat', 'ru':'логическое следствие', 'de':'Konsequenz', 'fr':'implication',
                    'nl':'gevolg'}

neutral_answers = {'en':'neutral', 'zh':'中立', 'ja':'中立', 'hi':'तटस्थ', 'sw':'upande wowote', 'bn':'নিরপেক্ষ',
                   'it':'neutro', 'no':'nøytral', 'es':'neutro',
                   'is':'hlutlaus', 'bg':'неутрален', 'sv':'neutral', 'sl':'nevtralen', 'ms':'neutral', 'th':'เป็นกลาง',
                   'pl':'neutralny', 'ru':'нейтральный', 'de':'neutral', 'fr':'neutre',
                   'nl':'neutrale'}

negative_answers = {'en':'contradiction', 'zh':'矛盾', 'ja':'矛盾', 'hi':'विरोधाभास', 'sw':'utata', 'bn':'দ্বন্দ্ব',
                    'it':'contraddizione', 'no':'motsigelse', 'es':'contradicción',
                    'is':'mótsögn', 'bg':'противоречие', 'sv':'motsägelse', 'sl':'protislovje', 'ms':'percanggahan', 'th':'ความขัดแย้ง',
                    'pl':'sprzeczność', 'ru':'противоречие', 'de':'Widerspruch', 'fr':'contradiction',
                    'nl':'tegenspraak'}


correct_num = 0

all_positive_out_ids = process_tokens(positive_answers['en'], tokenizer)
all_neutral_out_ids = process_tokens(neutral_answers['en'], tokenizer)
all_negative_out_ids = process_tokens(negative_answers['en'], tokenizer)
# print(all_positive_out_ids)
# print(all_neutral_out_ids)
# print(all_negative_out_ids)
# print(tokenizer('এনটেইলমেন্ট', return_tensors="pt"))
# print(tokenizer('দ্বন্দ্ব', return_tensors="pt"))

for step, inputs in enumerate(sample_iterator):
    # if step == 1:
    #      break
    # print(inputs)
    # add suffix for every instance in the batch
    if target_lang == 'en':
        input = [prompt_prefix['en'] + '\n' + "Premise: " + p + "\nHypothesis: " + h + '\nAnswer: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'zh':
        input = [prompt_prefix['zh'] + '\n' + "前提: " + p + "\n假设: " + h + '\n答案: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'ja':
        input = [prompt_prefix['ja'] + '\n' + "前提: " + p + "\n仮説: " + h + '\n答え: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'hi':
        input = [prompt_prefix['hi'] + '\n' + "आधार: " + p + "\nपरिकल्पना: " + h + '\nउत्तर: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'sw':
        input = [prompt_prefix['sw'] + '\n' + "Nguzo: " + p + "\nHypothesis: " + h + '\nJibu: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'bn':
        input = [prompt_prefix['bn'] + '\n' + "ভিত্তি: " + p + "\nহাইপোথিসিস: " + h + '\nউত্তর: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'it':
        input = [prompt_prefix['it'] + '\n' + "Premessa: " + p + "\nIpotesi: " + h + '\nRisposta: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'no':
        input = [prompt_prefix['no'] + '\n' + "Premiss: " + p + "\nHypotese: " + h + '\nSvar: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'es':
        input = [prompt_prefix['es'] + '\n' + "Premisa: " + p + "\nHipótesis: " + h + '\nRespuesta: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'is':
        input = [prompt_prefix['is'] + '\n' + "Forsenda: " + p + "\nTilgáta: " + h + '\nSvaraðu: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'bg':
        input = [prompt_prefix['bg'] + '\n' + "Предпоставка: " + p + "\nХипотеза: " + h + '\nОтговор: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'sv':
        input = [prompt_prefix['sv'] + '\n' + "Premiss: " + p + "\nHypotes: " + h + '\nSvar: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'sl':
        input = [prompt_prefix['sl'] + '\n' + "Predpogoj: " + p + "\nHipoteza: " + h + '\nOdgovori: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'ms':
        input = [prompt_prefix['ms'] + '\n' + "Premis: " + p + "\nHipotesis: " + h + '\nJawab: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'th':
        input = [prompt_prefix['th'] + '\n' + "ฐาน: " + p + "\nสมมติฐาน: " + h + '\nคำตอบ: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'pl':
        input = [prompt_prefix['pl'] + '\n' + "Przesłanka: " + p + "\nHipoteza: " + h + '\nOdpowiedź: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'ru':
        input = [prompt_prefix['ru'] + '\n' + "Предпосылка: " + p + "\nГипотеза: " + h + '\nОтвечать: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'de':
        input = [prompt_prefix['de'] + '\n' + "Prämisse: " + p + "\nHypothese: " + h + '\nAntwort: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'fr':
        input = [prompt_prefix['fr'] + '\n' + "Prémisse: " + p + "\nHypothèse: " + h + '\nRépondre: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    elif target_lang == 'nl':
        input = [prompt_prefix['nl'] + '\n' + "Premisse: " + p + "\nHypothese: " + h + '\nAntwoord: \"' for p, h in zip(inputs['premise'], inputs['hypothesis'])]
    
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
        # all_positive_out_ids = process_tokens(positive_answers[target_lang], tokenizer)
        # all_negative_out_ids = process_tokens(negative_answers[target_lang], tokenizer)
        # all_neutral_out_ids = process_tokens(neutral_answers[target_lang], tokenizer)
        all_positive_out_probs_sum = sum([probs[0][-1][i].item() for i in all_positive_out_ids])
        all_negative_out_probs_sum = sum([probs[0][-1][i].item() for i in all_negative_out_ids])
        all_neutral_out_probs_sum = sum([probs[0][-1][i].item() for i in all_neutral_out_ids])
        if all_positive_out_probs_sum > all_negative_out_probs_sum and all_positive_out_probs_sum > all_neutral_out_probs_sum:
            output = [input[0] + positive_answers['en']]
            # break
        elif all_negative_out_probs_sum > all_positive_out_probs_sum and all_negative_out_probs_sum > all_neutral_out_probs_sum:
            output = [input[0] + negative_answers['en']]
            # break
        elif all_neutral_out_probs_sum > all_positive_out_probs_sum and all_neutral_out_probs_sum > all_negative_out_probs_sum:
            output = [input[0] + neutral_answers['en']]
            # break
        else:
            if all_positive_out_probs_sum == all_negative_out_probs_sum and all_positive_out_probs_sum == all_neutral_out_probs_sum:
                print(f"WARNING: The model generates the same probability for all three choices!\n")
                rand = random.randint(0, 1)
                if rand >= (2/3):
                    output = [input[0] + positive_answers['en']]
                elif rand <= (1/3):
                    output = [input[0] + negative_answers['en']]
                else:
                    output = [input[0] + neutral_answers['en']]
            elif all_positive_out_probs_sum == all_negative_out_probs_sum:
                print(f"WARNING: The model generates the same probability for both positive and negative choices!\n")
                rand = random.randint(0, 1)
                if rand >= 0.5:
                    output = [input[0] + positive_answers['en']]
                else:
                    output = [input[0] + negative_answers['en']]
            elif all_positive_out_probs_sum == all_neutral_out_probs_sum:
                print(f"WARNING: The model generates the same probability for both positive and neutral choices!\n")
                rand = random.randint(0, 1)
                if rand >= 0.5:
                    output = [input[0] + positive_answers['en']]
                else:
                    output = [input[0] + neutral_answers['en']]
            elif all_negative_out_probs_sum == all_neutral_out_probs_sum:
                print(f"WARNING: The model generates the same probability for both negative and neutral choices!\n")
                rand = random.randint(0, 1)
                if rand >= 0.5:
                    output = [input[0] + neutral_answers['en']]
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
    # print(output)
    for i in range(len(output)):
        if (output[i].startswith(positive_answers['en'])):
            if (inputs['answer'][i] == positive_answers[target_lang]):
                correct_num += 1
        elif (output[i].startswith(neutral_answers['en'])):
            if (inputs['answer'][i] == neutral_answers[target_lang]):
                correct_num += 1
        elif (output[i].startswith(negative_answers['en'])):
            if (inputs['answer'][i] == negative_answers[target_lang]):
                correct_num += 1
        elif not (output[i].startswith(positive_answers['en']) or output[i].startswith(negative_answers['en']) or output[i].startswith(neutral_answers['en'])):
            print(f"Model generates an answer \"{output[i]}\" in the wrong format!\n")


print(f"\nTest set len is: {len(dataset)}")
print(f"Shot num is: 6")
print(f"Test model is: {model_name_or_path}\n")
print(f"Accuracy for {target_lang}: {correct_num / len(dataset)}")

