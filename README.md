# BertForTextGeneration
generating text with MLM(Masked Language Model)

# How to generate text?
Using [MASK] token to target tokens.  
  
for example, when input text is "The man is"  
STEP1  
  Input  : [CLS] The man [MASK]  
  Output : [CLS] The man sits.  
STEP2  
  Input  : [CLS] The man sits [MASK]  
  Output : [CLS] The man sits on  
...  
  
Above example is for illustration purposes only.  
Generated Text is terrible in the current version.  


# Generated Text(Using RoBerta)  
 
※mask_length = 10  

- Input          : The man eats  
  Generated Text : The man eats the chicken in the room</s>  
  
- Input          : My name is Clara and I am  
  Generated Text : My name is Clara and I am 18yr. and a young woman in South Carolina</s>
