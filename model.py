import sys
from typing import Literal

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

DEVICE = Literal['cpu', 'cuda']

class Generator():
    def __init__(self, 
                 model_name:str ="google/bert_uncased_L-12_H-768_A-12", 
                 device:DEVICE = 'cpu', 
                 eos_token:str = '[SEP]',
                 mask_token:str = '[MASK]') -> None:
        self.model_name = model_name
        try:
            self.mlm = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        except:
            print('model_name needs to be Masked Language Model(ex. google/bert_uncased_L-12_H-768_A-12)')
            print(f'but {model_name=}')
            sys.exit(1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.eos_token_id = self.tokenizer.encode(eos_token)[1]
        self.mask_token_id = self.tokenizer.encode(mask_token)[1]

    @torch.inference_mode()
    def generate(self, 
                 prefix:str, 
                 max_length:int = 64,
                 top_k = 3) -> str:
        tokenized_prefix = self.tokenizer(prefix, return_tensors='pt').to(self.device)
        generated_ids = tokenized_prefix.input_ids[0, 1:-1].tolist()
        max_length = max_length + len(generated_ids)
        while len(generated_ids) < max_length and generated_ids[-1] != self.eos_token_id:
            output = self.mlm(input_ids=torch.Tensor([generated_ids+[self.mask_token_id]]).long().to(self.device))
            generated_id = torch.topk(output.logits[0, -1], k=top_k).indices[torch.randint(low=0, high=top_k-1, size=(1,))]
            generated_ids.append(generated_id.item())
        decoded_text = self.tokenizer.decode(generated_ids)

        return decoded_text

if __name__ == '__main__':
    generator = Generator()
    text = ['The man eats',
            'A bird is',
            'Below you will find some great videos that will encourage you, train you and build you up in hearing from GOD and being able to let HIM fulfill HIS plan in your life. SOMETHING NEW THAT WILL HELP YOU HEAR GODS VOICE! How to']
    for t in text:
        print(generator.generate(t))
        print()