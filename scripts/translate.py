import argparse
import json

import torch
from transformers import FSMTTokenizer, FSMTForConditionalGeneration
from tqdm import tqdm


def translate_sentence(text, model, tokenizer, num_beams=10, **kwargs):
    device = model.device
    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        outputs = model.generate(input_ids, do_sample=True, top_p=0.95, **kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(
    input_path,
    output_path,
    lang_field,
    source_field,
    target_field
):
    EN_RU = "facebook/wmt19-en-ru"
    RU_EN = "facebook/wmt19-ru-en"
    en_ru_model = FSMTForConditionalGeneration.from_pretrained(EN_RU)
    en_ru_tokenizer = FSMTTokenizer.from_pretrained(EN_RU)
    ru_en_model = FSMTForConditionalGeneration.from_pretrained(RU_EN)
    ru_en_tokenizer = FSMTTokenizer.from_pretrained(RU_EN)
    mapping = {
        "ru": (ru_en_model, ru_en_tokenizer),
        "en": (en_ru_model, en_ru_tokenizer)
    }

    with open(input_path, "r") as r, open(output_path, "w") as w:
        for line in tqdm(r):
            record = json.loads(line)
            lang = record[lang_field]
            text = record[source_field]
            record[target_field] = translate_sentence(text, mapping[lang][0], mapping[lang][1])
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--lang-field", type=str, required=True)
    parser.add_argument("--source-field", type=str, required=True)
    parser.add_argument("--target-field", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
