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
