# Modelling-Indian-Names
Implemented different types of language models for modelling Indian names. There are clearly patterns in Indian names that models could learn, and we start modelling those using n-gram models, then move to neural n-gram and RNN models.

## Results:
## **Unigram:**

unigram train perplexity: 16.623900007096303

EVALUATION probability distribution is valid: True

EVALUATION of 1-gram on validation perplexity: 30.863827225768976

EVALUATION 1-gram generated names are o(oomm, jom,jm, m(, mhomoomd, o,mo

EVALUATION 1-gram generated names with prefix shjlmmm are shjlmmm, shjlmmm, shjlmmm, shjlmmm, shjlmmm

EVALUATION 1-gram top most likely chars after aa are a, i, n

### **Smoothed Unigram:**

smoothed unigram train perplexity: 16.623900007096303

EVALUATION probability distribution is valid: True

EVALUATION of 1-gram on validation perplexity: 11.14730834795356

EVALUATION 1-gram generated names are , dd, d, , [

EVALUATION 1-gram generated names with prefix shjlmmm are shjlmmm, shjlmmm, shjlmmm, shjlmmm, shjlmmm

EVALUATION 1-gram top most likely chars after aa are a, , , i, n

## **Bigram:**

bigram train perplexity: 7.658283554851139

EVALUATION probability distribution is valid: True

EVALUATION of 2-gram on validation perplexity: 5.594377295480409

EVALUATION 2-gram generated names are prana, shana, shaa, sh, bha

EVALUATION 2-gram generated names with prefix shjlmmm are shjlmmm, shjlmmm, shjlmmm, shjlmmm, shjlmmm

EVALUATION 2-gram top most likely chars after aa are , n, r, m, l

### **Smoothed Bigram:**

smoothed bigram train perplexity: 7.658283554851139

EVALUATION probability distribution is valid: True

EVALUATION of 2-gram on validation perplexity: 4.715609379404076

EVALUATION 2-gram generated names are n, am, gan, shana, aja

EVALUATION 2-gram generated names with prefix shjlmmm are shjlmmm, shjlmmm, shjlmmm, shjlmmm, shjlmmm

EVALUATION 2-gram top most likely chars after aa are , n, r, m, l

## **Smoothed Trigram:**

trigram train perplexity: 4.298966505162752

EVALUATION probability distribution is valid: True

EVALUATION of 3-gram on validation perplexity: 5.178486767038496

EVALUATION 3-gram generated names are aam, sam, sahi, san, sandeen

EVALUATION 3-gram generated names with prefix shjlmmm are shjlmmm, shjlmmm, shjlmmm, shjlmmm, shjlmmm

EVALUATION 3-gram top most likely chars after aa are n, s, r, l, m

## **Neural N-gram Language Model:**
### **FNN**
EVALUATION probability distribution is valid: True

EVALUATION of FNN on valid perplexity: 6.7446325639391134

EVALUATION RNN generated names are nanda, aocwno), shrewq, sumandana, ajsows

EVALUATION RNN generated names with prefix osmdfwo are coenfa, shjlmmmra, nsiao, asjdow, nssndao

EVALUATION RNN the top most likely chars after aa are v, r, m, n, l

## **Recurrent Neural Networks for Language Modelling**
### **RNN**
EVALUATION probability distribution is valid: True

EVALUATION of RNN on valid perplexity: 5.4496259689331055

EVALUATION RNN generated names are nanda, anaki), nabtil, sumandana, muetrMa

EVALUATION RNN generated names with prefix shjlmmm are shjlmmmay, shjlmmmra, shjlmmm, shjlmmmpa, shjlmmma

EVALUATION RNN the top most likely chars after aa are n, r, m, s, l

