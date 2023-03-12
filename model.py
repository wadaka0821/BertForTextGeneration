import sys
from typing import Literal

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

DEVICE = Literal['cpu', 'cuda']

class Generator():
    def __init__(self, 
                 model_name:str ="google/bert_uncased_L-12_H-768_A-12", 
                 device:DEVICE = 'cpu', 
                 eos_tokens:list[str] = ['[SEP]', '.'],
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
        self.eos_token_ids = [self.tokenizer.encode(t)[1] for t in eos_tokens]
        self.mask_token_id = self.tokenizer.encode(mask_token)[1]

    @torch.inference_mode()
    def generate(self, 
                 prefix:str, 
                 max_length:int = 64,
                 top_k = 10,
                 mask_length = 4) -> str:
        tokenized_prefix = self.tokenizer(prefix, return_tensors='pt').to(self.device)
        generated_ids = tokenized_prefix.input_ids[0, 1:-1].tolist()
        max_length = max_length + len(generated_ids)
        while len(generated_ids) < max_length and generated_ids[-1] not in self.eos_token_ids:
            output = self.mlm(input_ids=torch.Tensor([generated_ids+[self.mask_token_id for _ in range(mask_length)]]).long().to(self.device))
            generated_id = torch.topk(output.logits[0, -mask_length], k=top_k).indices[torch.randint(low=0, high=top_k-1, size=(1,))]
            generated_ids.append(generated_id.item())
        decoded_text = self.tokenizer.decode(generated_ids)

        return decoded_text

if __name__ == '__main__':
    generator = Generator(model_name="xlm-roberta-base", eos_tokens=['</s>', '.'], mask_token='<mask>')
    text = ['The man eats',
            'A bird is',
            'My name is Clara and I am']
    for t in text:
        print(generator.generate(t))
        print()