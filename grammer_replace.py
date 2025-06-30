import json
import re

def replace_terms(sentence):
    sentence = re.sub(r'\bPedestrians\b', 'Persons', sentence)
    sentence = re.sub(r'\bpedestrians\b', 'persons', sentence)
    sentence=re.sub(r'\bpedestrain\b', 'person', sentence)
    sentence = re.sub(r'\bmotorcycles\b', 'motorbikes', sentence)
    sentence = re.sub(r'\bmotorcycle\b', 'motorbike', sentence)
    sentence = re.sub(r'\bMotorcycles\b', 'Motorbike', sentence)
    return sentence
def process_json(json_data):
    """处理 JSON 数据中的每个句子"""
    for key, sentences in json_data.items():
        corrected_sentences = []
        for sentence in sentences:
            corrected_sentence = replace_terms(sentence)
            corrected_sentences.append(corrected_sentence)
        json_data[key] = corrected_sentences
    return json_data

# 读取 JSON 文件
input_file = '/data/dada_new.json'
output_file = '/data/dada_new_new.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理 JSON 数据
corrected_data = process_json(data)

# 保存到新的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(corrected_data, f, ensure_ascii=False, indent=4)

print(f"save {output_file}")