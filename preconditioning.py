"""
预处理数据
将文本数据进行拆分为句子，随机替换其中 部分，进行判断处理。
"""
import re
import json
import random
import csv

def text2data(text):
    """
    将文本转换成数据集

    替换第六个句子

    :param text:
    :return:
    """
    out = re.split(r"[。|！|?|？|！|,|，|：|\n|\r]", text)
    # print(out)
    if len(out) > 10:
        # print(out)
        for i in range(len(out) - 10):

            sents = out[i:i + 10]
            yield {"sents": sents, "label": "无冗余"}

            replace_sent = random.choices(out)[0]
            sents[5] = replace_sent
            if replace_sent == sents[3]:
                pass
            else:
                sents=sents[:4]+[replace_sent]+sents[4:]
                # print(len(sents[:10]))
                yield {"sents": sents[:10], "label": "冗余"}

outFile=open("data/data.csv",'w')
witer=csv.DictWriter(outFile,fieldnames=["sent","label","topic"])
witer.writeheader()
with open("data/web_text_zh_testa.json",'r') as f:
    for i,it in enumerate(f):
        # print(it)
        data=json.loads(it)
        text=data['content']
        for item in text2data(text):
            # print(item)
            if len("".join(item["sents"]))>10:
                witer.writerow({"sent":" [SEP] ".join(item["sents"]),"label":item['label'],"topic":data["topic"]})
        # if i>10:
        #     break
outFile.close()