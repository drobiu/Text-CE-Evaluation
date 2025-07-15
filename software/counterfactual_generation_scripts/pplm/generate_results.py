import json
import run_pplm
import torch
import tqdm

import numpy as np

from datasets import load_dataset

def get_text_conditioning(text):
        min_n = 10
        
        words = text.split(" ")
        
        n = int(len(words)/2)
        
        words = words[:max(min_n, n)]
        
        return " ".join(words)

if __name__ == "__main__":
    print(f"CUDA ready: {torch.cuda.is_available()}")

    classifier_fomc = "gtfintechlab/FOMC-RoBERTa"

    dataset_id = "TextCEsInFinance/fomc-communication-counterfactual"

    dataset = load_dataset(dataset_id)

    test = dataset['test'].to_pandas()[['text', 'label', 'text_label', 'target']]

    test = test.assign(conditioning = test['text'].apply(lambda x: get_text_conditioning(x)))
    results = []

    for i in tqdm.tqdm(range(len(test))):
        # the base sentence
        row = test.iloc[i]
        cond_text = row['conditioning']
        target = row['target']

        perturbations = run_pplm.run_pplm_example(
            pretrained_model="gpt2-medium",
            cond_text=cond_text,
            num_samples=5,
            discrim="generic",
            discrim_weights="generic_classifier_head_epoch_9.pt",
            discrim_meta="generic_classifier_head_meta.json",
            class_label=target,
            length=25,
            stepsize=0.01,
            gamma=1.0,
            gm_scale=0.9,
            kl_scale=0.01,
            verbosity='quiet')
        
        results.append({
            'id': i + 1,
            'text': row['text'],
            'label': row['label'],
            'cond_text': cond_text,
            'counterfactuals': perturbations,
            'target': target,
        })

    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()

    with open('results.json', 'w') as file:
        file.write(json.dumps(results, default=np_encoder, indent=4))
