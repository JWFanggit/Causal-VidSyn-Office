import json
import language_tool_python

# 初始化 LanguageTool
tool = language_tool_python.LanguageTool('en-US')

def check_and_correct_sentence(sentence):
    """检查并修改输入的句子"""
    matches = tool.check(sentence)
    corrected_sentence = language_tool_python.utils.correct(sentence, matches)
    return corrected_sentence

def process_json(json_data):
    """处理 JSON 数据中的每个句子"""
    for key, sentences in json_data.items():
        corrected_sentences = []
        for sentence in sentences:
            corrected_sentence = check_and_correct_sentence(sentence)
            corrected_sentences.append(corrected_sentence)
        json_data[key] = corrected_sentences
    return json_data

# 读取 JSON 文件
input_file = '/data/dada.json'
output_file = '/data/dada_new.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理 JSON 数据
corrected_data = process_json(data)

# 保存到新的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(corrected_data, f, ensure_ascii=False, indent=4)

print(f"语法检查和修正完成，保存到 {output_file}")