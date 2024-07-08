# Modelling-Indian-Names
Implemented different types of language models for modelling Indian names. There are clearly patterns in Indian names that models could learn, and we start modelling those using n-gram models, then move to neural n-gram and RNN models.

```
Modelling-Indian-Names
├── Modelling-Indian-Names.ipynb
├── main.py
├── fnn
│   ├── loss.json
│   ├── model.pt
│   └── vocab.pt
├── rnn
│   ├── loss.json
│   ├── model.pt
│   └── vocab.pt
└── README.md
```

## N-gram language models
In natural language processing, an n-gram is a sequence of n words. For example, “statistics” is a unigram (n = 1), “machine learning” is a bigram (n = 2), “natural language processing” is a trigram (n = 3). For longer n-grams, people just use their lengths to identify them, such as 4-gram, 5-gram, and so on.

![image](https://github.com/pritamgouda11/Modelling-Indian-Names/assets/46958858/5a1398d0-9029-48fb-be66-9d7dbc63365f)

## N-Grams Smoothing
The standard N-gram models are trained from some corpus. The finiteness of the training corpus leads to the absence of some perfectly acceptable N-grams. This results in sparse bigram matrices. This method tend to underestimate the probability of strings that do not occur in their training corpus.

There are some techniques that can be used for assigning a non-zero probabilty to these 'zero probability bigrams'. This task of reevaluating some of the zero-probability and low-probabilty N-grams, and assigning them non-zero values, is called smoothing. Some of the techniques are: Add-One Smoothing, in Add-One smooting, we add one to all the bigram counts before normalizing them into probabilities. This is called add-one smoothing.

<img width="1238" alt="Screenshot 2024-06-29 at 12 02 27 AM" src="https://github.com/pritamgouda11/Modelling-Indian-Names/assets/46958858/47c525d1-dc50-402e-97fe-702147bc4590">

## Results: 

### **Unigram:**
```
unigram train perplexity: 16.623900007096303
EVALUATION of 1-gram on validation perplexity: 30.863827225768976
EVALUATION 1-gram top most likely chars after aa are a, i, n
```
### **Smoothed Unigram:**
```
smoothed unigram train perplexity: 16.623900007096303
EVALUATION of 1-gram on validation perplexity: 11.14730834795356
EVALUATION 1-gram top most likely chars after aa are a, r, i, n
```
### **Bigram:**
```

bigram train perplexity: 7.658283554851139
EVALUATION of 2-gram on validation perplexity: 5.594377295480409
EVALUATION 2-gram generated names are prana, shana, shaa, sh, bha
EVALUATION 2-gram top most likely chars after aa are , n, r, m, l
```
### **Smoothed Bigram:**
```

smoothed bigram train perplexity: 7.658283554851139
EVALUATION of 2-gram on validation perplexity: 4.715609379404076
EVALUATION 2-gram top most likely chars after aa are , n, r, m, l
```
### **Smoothed Trigram:**
```

trigram train perplexity: 4.298966505162752
EVALUATION of 3-gram on validation perplexity: 5.178486767038496
EVALUATION 3-gram generated names are aam, sam, sahi, san, sandeen
EVALUATION 3-gram top most likely chars after aa are n, s, r, l, m
```
## **Neural N-gram Language Model:**

**Feed-Forward Neural Networks**
The feedforward neural network is one of the most basic artificial neural networks. In this ANN, the data or the input provided travels in a single direction. It enters into the ANN through the input layer and exits through the output layer while hidden layers may or may not exist. So the feedforward neural network has a front-propagated wave only and usually does not have backpropagation

**Recurrent Neural Networks**
The Recurrent Neural Network saves the output of a layer and feeds this output back to the input to better predict the outcome of the layer. The first layer in the RNN is quite similar to the feed-forward neural network and the recurrent neural network starts once the output of the first layer is computed. After this layer, each unit will remember some information from the previous step so that it can act as a memory cell in performing computation

![image](https://github.com/pritamgouda11/Modelling-Indian-Names/assets/46958858/80a8e2ba-47f6-48d2-8b09-0138c0f63809)

### **FNN**
```

EVALUATION of FNN on valid perplexity: 6.7446325639391134
EVALUATION RNN generated names are nanda, shrewq, sumandana, ahas
EVALUATION RNN the top most likely chars after aa are v, r, m, n, l
```
### **RNN**
```

EVALUATION of RNN on valid perplexity: 5.4496259689331055
EVALUATION RNN generated names are nanda, anaki, nabtil, sumandana, mytri
EVALUATION RNN the top most likely chars after aa are n, r, m, s, l
```
