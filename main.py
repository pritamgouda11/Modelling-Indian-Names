import pandas as pd
import numpy as np
import math
import copy
import random
from collections import Counter, defaultdict

from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

"""Please note that we may change the contents of the following four files when we rerun your code, so please make sure that your solution is not specifically engineered to just these names."""

# Download the training and validation datasets
!wget -O train_data.csv "https://docs.google.com/spreadsheets/d/1AUzwOQQbAehg_eoAMCcWfwSGhKwSAtnIzapt2wbv0Zs/gviz/tq?tqx=out:csv&sheet=train_data.csv"
!wget -O valid_data.csv "https://docs.google.com/spreadsheets/d/1UtQErvMS-vcQEwjZIjLFnDXlRZPxgO1CU3PF-JYQKvA/gviz/tq?tqx=out:csv&sheet=valid_data.csv"

# Download the text for evaluation
!wget -O eval_prefixes.txt "https://drive.google.com/uc?export=download&id=1tuRLJXLd2VcDaWENr8JTZMcjFlwyRo60"
!wget -O eval_sequences.txt "https://drive.google.com/uc?export=download&id=1kjPAR04UTKmdtV-FJ9SmDlotkt-IKM3b"

## Please do not change anything in this code block.

def read_dataframe(ds_type):
    """
    Args:
        ds_type [str] :  dataset type (train or valid)

    Returns:
        df [pandas dataframe]
    """

    df = pd.read_csv(f"/content/{ds_type}_data.csv", header=0, index_col=0)
    df = df[~df['Name'].isna()]
    df['Name'] = df['Name'].astype(str)
    return df

# Load the training and validation datasets
train_data = read_dataframe("train")
validation_data = read_dataframe("valid")

# Read files containing prefixes and character sequences for evaluation
with open('eval_prefixes.txt', 'r') as file:
    eval_prefixes = []
    for line in file:
        eval_prefixes.append(line.strip().split(" "))

with open('eval_sequences.txt', 'r') as file:
    eval_sequences = []
    for line in file:
        eval_sequences.append(line.strip().split(" "))

print(f"Length of training data: {len(train_data)}\nLength of validation data: {len(validation_data)}")

## Please do not change anything in this code block.

START = "<s>"   # Start-of-name token
END = "</s>"    # End-of-name token
UNK = "<unk>"   # token representing out of unknown (or out of vocabulary) tokens
vocab_from_ascii = True

def build_vocab(names):
    """
    Builds a vocabulary given a list of names

    Args:
        names [list[str]]: list of names

    Returns:
        vocab [torchtext.vocab]: vocabulary based on the names

    """

    if vocab_from_ascii:
        char_counts = {chr(i):i for i in range(128)}
    else:
        char_counts = Counter("".join(names))

    vocab = build_vocab_from_iterator(
                    char_counts,
                    specials=[UNK, START, END], #adding special tokens to the vocabulary
                    min_freq=1
                )
    vocab.set_default_index(vocab[UNK])
    return vocab


def tokenize_name(name):
    """
    Tokenise the name i.e. break a name into list of characters

    Args:
        name [str]: name to be tokenized

    Returns:
        list of characters
    """

    return list(str(name))


def process_data_for_input(data_iter, vocab):
    """
    Processes data for input: Breaks names into characters,
    converts out of vocabulary tokens to UNK and
    appends END token at the end of every name

    Args:
        data_iter: data iterator consisting of names
        vocab: vocabulary

    Returns:
        data_iter [list[list[str]]]: list of names, where each name is a
                                list of characters and is appended with
                                START and END tokens

    """

    vocab_set = set(vocab.get_itos())
    # convert Out Of Vocabulary (OOV) tokens to UNK tokens
    data_iter = [[char if char in vocab_set else UNK
                        for char in tokenize_name(name)] for name in data_iter]
    data_iter = [[START] + name + [END] for name in data_iter]

    return data_iter


def get_tokenised_text_and_vocab(ds_type, vocab=None):
    """
    Reads input data, tokenizes it, builds vocabulary (if unspecified)
    and outputs tokenised list of names (which in turn is a list of characters)

    Args:
        ds_type [str]: Type of the dataset (e.g., train, validation, test)
        vocab [torchtext.vocab]: vocabulary;
                                 If vocab is None, the function will
                                 build the vocabulary from input text.
                                 If vocab is provided, it will tokenize name
                                 according to the vocab, replacing any tokens
                                 not part of the vocab with UNK token.

    Returns:
        data_iter: data iterator for tokenized names
        vocab: vocabulary

    """

    # read the 'Name' column of the dataframe
    if ds_type=='train':
        data_iter = train_data['Name']
    elif ds_type=='valid':
        data_iter = validation_data['Name']
    else:
        data_iter = test_data['Name']

    # build vocab from input data, if vocab is unspecified
    if vocab is None:
        vocab = build_vocab(data_iter)

    # convert OOV chars to UNK, append START and END token to each name
    data_iter = process_data_for_input(data_iter, vocab)

    return data_iter, vocab

"""Let's look at some examples from the training set"""

# Look at some random examples from the training set
examples = ", ".join(random.sample(list(train_data['Name']), 5))
f"Examples from the training set: {examples}"

"""# Module 1: N-gram Language Modelling

Load and preprocess the data for n-gram models
"""

"""choose your hyperparameter and see the difference in performance"""

# CHANGE THE None VALUES TO YOUR DESIRED VALUES

# ADD YOUR CODE HERE
#BEGIN CODE
names = train_data['Name']
name_lengths = names.str.len()
max_length = name_lengths.max()
mean_length = name_lengths.mean()
print("Mean length of names:", mean_length)
print("Maximum length of names:", max_length)
# I am choosing it long enough to include the majority of the names while excluding outliers
percentile_length = name_lengths.quantile(0.95)
print("95th percentile length:", percentile_length)
# END CODE

MAX_NAME_LENGTH = 9 # maximum length of names for generation

# Get data iterator and build vocabulary from input text

train_text, vocab = get_tokenised_text_and_vocab(ds_type='train')
validation_text, _ = get_tokenised_text_and_vocab(ds_type='valid', vocab=vocab)

# Check the size of vocabulary
vocab_size = len(vocab.get_stoi())
print(vocab_size)

# BEIGN CODE
total_tokens_train = sum(len(text) for text in train_text)
total_tokens_valid = sum(len(text) for text in validation_text)
print(f"Total number of tokens in training set: {total_tokens_train}")
print(f"Total number of tokens in validation set: {total_tokens_valid}")
# END CODE
def get_unigram_counts(corpus):
    """
    Given a corpus, calculates the unigram counts for each character in the corpus

    Args:
        corpus [list[list[str]]]: list of tokenized characters. Text is appended with END token.

    Returns:
        unigram_counts [dict [key: char, value: count]]:
            dictionary of unigram counts for each character in the corpus
        Example:
        > unigram_counts["c1"] = 5
    """

     # ADD YOUR CODE HERE
     # BEGIN CODE
    unigram_counts ={}
    for names in corpus:
        for name in names:
            if name in unigram_counts:
                unigram_counts[name] = unigram_counts[name]+1
            else:
                unigram_counts[name]=1
      # END CODE
    return unigram_counts

def get_bigram_counts(corpus):
    """
    Given a corpus, calculates the bigram counts for each bigram in the corpus.
    The corpus *only* contains END tokens at the end of names.
    You may want to handle the case whhere beginning of the name
    does not have n-1 prior chars.

    Args:
        corpus [list[list[str]]]: list of tokenized text. Text is appended with END token.

    Returns:
        bigram_counts [dict[dict]]:
            nested dictionary of bigram counts for each bigram in the corpus
        Example:
        > bigram_counts["c1"]["c2"] = 5
        here bigram_counts["c1"]["c2"] represents P("c2"|"c1")
        P["c1"]["c2"] means P[char_i = "c2"|char_{i-1} = "c1"]
    """

    # ADD YOUR CODE HERE
    # BEGIN CODE
    bigram_counts = defaultdict(lambda: defaultdict(int))
    for name in corpus:
        for i in range(len(name) - 1):
            first_char = name[i]
            second_char = name[i + 1]
            bigram_counts[first_char][second_char] += 1
    for char, followers in bigram_counts.items():
        if not followers:
            bigram_counts[char]['<unk>'] = 1
    bigram_counts = {char: dict(followers) for char, followers in bigram_counts.items()}
    # END CODE
    return bigram_counts

def get_trigram_counts(corpus):
    """
    Given a corpus, calculates the trigram counts for each trigram in the corpus.
    The corpus *only* contains END tokens at the end of names.
    You may want to handle the case where beginning of the text
    does not have n-1 prior chars.

    Args:
        corpus [list[list[str]]]: list of tokenized text. Text is appended with END token.

    Returns:
        trigram_counts [dict[dict[dict]]]:
            nested dictionary for each trigram in the corpus
        Example:
        > trigram_counts["c1"]["c2"]["c3"] = 5
        P["c1"]["c2"]["c3] means P[char_i = "c3"|char_{i-2} = "c1", char_{i-1} = "c2"]

    """

    # ADD YOUR CODE HERE
    # BEGIN CODE
    trigram_counts = {}
    for text in corpus:
        for position in range(len(text) - 2):
            first_char, second_char, third_char = text[position], text[position + 1], text[position + 2]
            if first_char not in trigram_counts:
                trigram_counts[first_char] = {second_char: {third_char: 1}}
            elif second_char not in trigram_counts[first_char]:
                trigram_counts[first_char][second_char] = {third_char: 1}
            else:
                trigram_counts[first_char][second_char].setdefault(third_char, 0)
                trigram_counts[first_char][second_char][third_char] += 1
      # END CODE
    return trigram_counts

"""
Implementation of the n-gram language models.
All other n-gram models (unigram, bigram, etc.) would follow the same skeleton.
"""

class NGramLanguageModel(object):
    def __init__(self, train_text):
        """
        Initialise and train the model with train_text.

        Args:
            train_text [list of list]: list of tokenised names

        Returns:
            -
        """
        return

    def get_next_char_probabilities(self):
        """
        Returns a probability distribution over all chars in the vocabulary.
        Probability distribution should sum to one.

        Returns:
            P: dictionary or nested dictionary; Output format depends on n-gram
            Examples:
                for N=1 (unigram); dict[key:unigram,value:probability of unigram]
                    > P["c1"] = 0.0001
                for N=2 (bigram); dict[key:bigram_char1, value:dict[key:bigram_char2,value:probability of bigram]]
                    > P["c1"]["c2"] = 0.0001
                    P["c1"]["c2"] means P["c2"|"c1"]
                for N=3 (trigram); dict[dict[dict]]
                    > P["c1"]["c2"]["c3"] = 0.0001
                    P["c1"]["c2"]["c3] means P[char_i = "c3"|char_{i-2} = "c1", char_{i-1} = "c2"]
        """

        return


    def get_name_log_probability(self, name):
        """
        Calculates the log probability of name according to the language model

        Args:
            name [list]: list of tokens

        Returns:
            log_prob [float]: Log probability of the given name
        """
        return 0.0


    def get_perplexity(self, text):
        """
        Returns the perplexity of the model on a text as a float.

        Args:
            text [list]: a list of string tokens

        Returns:
            perplexity [float]: perplexity of the given text
        """
        return 0.0


    def generate_names(self, k, n=MAX_NAME_LENGTH, prefix=None):
        """
        Given a prefix, generate k names according to the model.
        The default prefix is None.
        You may stop the generation when n tokens have been generated,
        or when you encounter the END token.

        Args:
            k [int]: Number of names to generate
            n [int]: Maximum length (number of tokens) in the generated name
            prefix [list of tokens]: Prefix after which the names have to be generated

        Returns:
            names [list[str]]: list of generated names
        """
        return []

    def get_most_likely_chars(self, sequence, k):
        """
        Given a sequence of characters, outputs k most likely characters after the sequence.

        Args:
            sequence [list[str]]: list of characters
            k [int]: number of most likely characters to return

        Returns:
            chars [list[str]]: *Ordered* list of most likely characters
                        (with charcater at index 0 being the most likely and
                        character at index k-1 being the least likely)

        """
        return []

## Please do not change anything in this code block.

def check_validity(model, ngram, is_neural):
    """
    Checks if get_next_char_probabilities returns a valid probability distribution
    """

    if ngram==1 or is_neural:
        P = model.get_next_char_probabilities()
        is_valid = validate_probability_distribution(P.values())
        if not is_valid:
            return is_valid

    elif ngram==2:
        P = model.get_next_char_probabilities()
        for char1 in P.keys():
            is_valid = validate_probability_distribution(list(P[char1].values()))
            if not is_valid:
                return is_valid

    elif ngram==3:
        P = model.get_next_char_probabilities()
        for char1 in P.keys():
            for char2 in P[char1].keys():
                is_valid = validate_probability_distribution(list(P[char1][char2].values()))
                if not is_valid:
                    return is_valid
    else:
        print("Enter a valid number for ngram")

    return True


def validate_probability_distribution(probs):
    """
    Checks if probs is a valid probability distribution
    """
    if not min(probs) >= 0:
        print("Negative value in probabilities")
        return False
    elif not max(probs) <= 1 + 1e-8:
        print("Value larger than 1 in probabilities")
        return False
    elif not abs(sum(probs)-1) < 1e-4:
        print("probabilities do not sum to 1")
        return False
    return True


def eval_ngram_model(model, ngram, ds, ds_name, eval_prefixes, eval_sequences, num_names=5, is_neural=False):
    """
    Runs the following evaluations on n-gram models:
    (1) checks if probability distribution returned by model.get_next_char_probabilities() sums to one
    (2) checks the perplexity of the model
    (3) generates names using model.generate_names()
    (4) generates names given a prefix using model.generate_names()
    (4) output most likely characters after a given sequence of chars using model.get_most_likely_chars()
    """

    # (1) checks if probability distributions sum to one
    is_valid = check_validity(model=model, ngram=ngram, is_neural=is_neural)
    print(f'EVALUATION probability distribution is valid: {is_valid}')

    # (2) evaluate the perplexity of the model on the dataset
    print(f'EVALUATION of {ngram}-gram on {ds_name} perplexity:',
        model.get_perplexity(ds))

    # (3) generate a few names
    generated_names = ", ".join(model.generate_names(k=num_names))
    print(f'EVALUATION {ngram}-gram generated names are {generated_names}')

    # (4) generate a few names given a prefix
    for prefix in eval_prefixes:
        generated_names_with_prefix = ", ".join(model.generate_names(k=num_names, prefix=prefix))
        prefix = ''.join(prefix)
        print(f'EVALUATION {ngram}-gram generated names with prefix {prefix} are {generated_names_with_prefix}')

    # (5) get most likely characters after a sequence
    for sequence in eval_sequences:
        most_likely_chars = ", ".join(model.get_most_likely_chars(sequence=sequence, k=num_names))
        sequence = "".join(sequence)
        print(f"EVALUATION {ngram}-gram top most likely chars after {sequence} are {most_likely_chars}")

"""## 1.1 Unigram"""

"""
Implementaion of a Unigram Model without smoothing
"""

class UnigramModel(NGramLanguageModel):
    def __init__(self, train_text):
        """
        Initialise and train the model with train_text.

        Args:
            train_text [list of list]: list of tokenised names
        """

        # ADD YOUR CODE HERE
        #BEGIN CODE
        self.training_corpus= train_text
        self.unigram_counts = get_unigram_counts(train_text)
        self.vocabulary = set(self.unigram_counts.keys())
        #END CODE

    def get_next_char_probabilities(self):
        """
        Return a dictionary of probabilities for each char in the vocabulary

        Returns:
            key: char, value: probability
        """

        # ADD YOUR CODE HERE
        #BEGIN CODE
        total_chars = sum(self.unigram_counts.values())
        return {char: freq / total_chars for char, freq in self.unigram_counts.items()}
        #END CODE

    def get_name_log_probability(self, name):
        """
        Calculates the log probability of name according to the n-gram model

        Args:
            name [list]: list of tokens

        Returns:
            Log probability of the name [float]
        """

        # ADD YOUR CODE HERE
        # BEGIN CODE
        probabilities = self.get_next_char_probabilities()
        log_probability = 0.0
        for char in name:
            log_probability += np.log(probabilities.get(char, 1e-6))
        # END CODE
        return log_probability


    def get_perplexity(self, text):
        """
        Returns the perplexity of the model on a text as a float.

        Args:
            text [list]: a list of string tokens

        Returns:
            perplexity of the given text [float]
        """

        # ADD YOUR CODE HERE
        # BEGIN CODE
        log_probability = 0.0
        char_count = 0
        for name in text:
            log_probability += self.get_name_log_probability(name)
            char_count += len(name)
        # END CODE
        return math.exp(-log_probability / char_count)



    def select_next_char(self, name, temperature=1.0):
        probabilities = np.array(list(self.get_next_char_probabilities().values()))
        adjusted_probs = np.power(probabilities, 1 / temperature) / np.sum(np.power(probabilities, 1 / temperature))
        return np.random.choice(list(self.vocabulary), p=adjusted_probs)

    def generate_names(self, k, n=MAX_NAME_LENGTH, prefix=None):
        """
        Given a prefix, generate k names according to the model.
        The default prefix is None.

        Args:
            k [int]: Number of names to generate
            n [int]: Maximum length (number of tokens) in the generated name
            prefix [list of tokens]: Prefix after which the names have to be generated

        Returns:
            list of generated names [list]
        """
        # ADD YOUR CODE HERE
        # BEGIN CODE
        names = []
        for _ in range(k):
            if prefix:
                if prefix[-1] == '</s>':
                    prefix =prefix[:-1]
                name=prefix
            else:
                name=['<s>']
            while len(name) <n and name[-1] != "</s>":
                next_char = self.select_next_char(name,0.5)
                if next_char == "<s>" and len(name) > 1:
                    next_char = self.select_next_char(name,0.5)
                name.append(next_char)
            names.append(name)
        new_names = [''.join(sublist) for sublist in names]
        # END CODE
        return new_names

    def get_most_likely_chars(self, sequence, k):
        """
        Given a sequence of characters, outputs k most likely characters after the sequence.

        Args:
            sequence [list[str]]: list of characters
            k [int]: number of most likely characters to return

        Returns:
            chars [list[str]]: *Ordered* list of most likely characters
                        (with charcater at index 0 being the most likely and
                        character at index k-1 being the least likely)

        """
        # ADD YOUR CODE HERE
        char_probabilities = self.get_next_char_probabilities()
        sorted_probs = sorted(char_probabilities.items(), key=lambda x: x[1], reverse=True)
        return [char for char, _ in sorted_probs[:k]]

"""### Eval

**Note**: For models without smoothing, you may observe perplexity as `inf` if the validation or test set contains characters not seen in the train set
However, this should not happen for models where you implement smoothing.
"""

## Please do not change anything in this code block.

unigram_model = UnigramModel(train_text)

# Check the perplexity of the unigram model on the train set
print('unigram train perplexity:',
      unigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=unigram_model, ngram=1, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

"""### Smoothing

Implement a smoothed version of the unigram model. You may extend the `UnigramModel` class and re-use some of the functions.  For unigram model, you should implement Add-1 smoothing.

You may refer to the lecture slides or [3.5 Smoothing](https://web.stanford.edu/~jurafsky/slp3/3.pdf) for details on different smoothing technqiues.

"""

"""
Implementation of unigram model with Add-1 smoothing.

"""
class SmoothedUnigramModel(UnigramModel):

    def __init__(self, train_text):
        super().__init__(train_text)

    # You should override ONLY those functions
    # which calculate probability of a unigram.
    # You can override get_next_char_probabilities
    # or any other helper functions you use in UnigramModel
    # to calculate unigram probabilities.

    # Implement Laplace or Add-1 smoothing for the unigram model

    # ADD YOUR CODE HERE
    def get_name_log_probability(self, name):
        char_probs = self.get_next_char_probabilities()
        log_prob =0.0
        total = sum(self.unigram_counts.values())
        for char in name:
            if char in self.vocabulary:
                log_prob += np.log(char_probs[char])
            else:
                log_prob += np.log(1 / total + len(self.vocabulary))
        return log_prob

## Please do not change anything in this code block.

smoothed_unigram_model = SmoothedUnigramModel(train_text)

# Check the perplexity of the smoothed unigram model on the train set
print('smoothed unigram train perplexity:',
      smoothed_unigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=smoothed_unigram_model, ngram=1, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences,  num_names=5)

# Release models we don't need any more.
del unigram_model
del smoothed_unigram_model

"""## 1.2 Bigram"""



class BigramModel(NGramLanguageModel):
    # BEGIN CODE
    def calculate_char_probabilities(self, bigram_counts):
        char_probs = {}
        outer_key_sums = {key: sum(value.values()) for key, value in bigram_counts.items()}

        for outer_key, inner_dict in bigram_counts.items():
            char_probs[outer_key] = {inner_key: value / outer_key_sums[outer_key] for inner_key, value in inner_dict.items()}

        return char_probs
    # END CODE

    def __init__(self, train_text):
        """
        Initialize and train the model with train_text.

        Args:
            train_text [list of list]: List of tokenized names
        """
        # BEGIN CODE
        self.train_text = train_text
        self.bigram_counts = get_bigram_counts(train_text)
        int_iter = dict()
        for key, value in self.bigram_counts.items():
            int_iter[key] = set(value.keys())
        self.vocabulary = int_iter
        self.char_probs = self.calculate_char_probabilities(self.bigram_counts)
        # END CODE

    def get_next_char_probabilities(self):
        """
        Returns a probability distribution over all characters in the vocabulary.
        Probability distribution should sum to one.

        Returns:
            P: Probability distribution over characters
        """
        # BEGIN CODE
        return self.char_probs
        # END CODE

    def get_name_log_probability(self, name):
        """
        Calculates the log probability of name according to the bigram model.

        Args:
            name [list]: List of tokens (characters)

        Returns:
            Log probability of the name [float]
        """
        # ADD YOUR CODE
        # BEGIN CODE
        log_prob = 0.0
        for i in range(len(name) - 1):
            char1, char2 = name[i], name[i + 1]
            if char1 in self.char_probs:
                if char2 in self.char_probs[char1]:
                    log_prob += np.log(self.char_probs[char1].get(char2, 0))
        # END CODE
        return log_prob

    def get_perplexity(self, text):
        """
        Returns the perplexity of the model on a text as a float.

        Args:
            text [list]: List of string tokens

        Returns:
            perplexity of the given text [float]
        """
        # ADD YOUR CODE
        # BEGIN CODE
        log_prob = 0.0
        n = 0
        for name in text:
            log_prob += self.get_name_log_probability(name)
            n += len(name)

        perplexity = math.exp(-log_prob / n)
        # END CODE
        return perplexity

    def sample_next_char (self,name, temperature):
        # BEGIN CODE
        prev_char = name[-1]
        next_char_probs = self.char_probs.get(prev_char, {})
        if temperature > 0.0:
            probs = [p ** (1 / temperature) for p in next_char_probs.values()]
            sum_probs = sum(probs)
            probs = [p / sum_probs for p in probs]
        else:
            probs = next_char_probs.values()
        next_char = random.choices(list(next_char_probs.keys()), weights=probs)[0]
        if next_char not in self.vocabulary[prev_char]:
            next_char = random.choice(list(self.vocabulary[prev_char]))
        # END CODE
        return next_char

    def generate_names(self, k, n=MAX_NAME_LENGTH, prefix=None):
        """
        Generates k names according to the model.

        Args:
            k [int]: Number of names to generate
            n [int]: Maximum length (number of tokens) in the generated name
            prefix [list of tokens]: Prefix after which the names have to be generated

        Returns:
            generated_names [list]: List of generated names
        """
        # ADD YOUR CODE
        # BEGIN CODE
        generated_names = []
        for _ in range(k):
            name = prefix if prefix else ['<s>']
            while len(name) < n and name[-1] != '</s>':
                next_char = self.sample_next_char(name, 0.5)
                if next_char == "<unk>":
                    unigram_model = UnigramModel(train_text)

                    next_char = self.unigram_model.sample_next_char(name,0.5)
                name.append(next_char)
            generated_names.append(''.join(name))
        # END CODE
        return generated_names

    def get_most_likely_chars(self, sequence, k):
        """
        Given a sequence of characters, outputs k most likely characters after the sequence.

        Args:
            sequence [list[str]]: List of characters
            k [int]: Number of most likely characters to return

        Returns:
            most_likely_chars [list[str]]: Ordered list of most likely characters
        """
        # BEGIN CODE
        last_char = sequence[-1]
        next_char_probs = self.char_probs.get(last_char, {})
        valid_chars = self.vocabulary[last_char]
        next_char_probs = {char: prob for char, prob in next_char_probs.items() if char in valid_chars}
        sorted_chars = sorted(next_char_probs, key=next_char_probs.get, reverse=True)
        most_likely_chars = sorted_chars[:k]
        # END CODE
        return most_likely_chars

"""### Eval"""

## Please do not change anything in this code block.

bigram_model = BigramModel(train_text)

# check the perplexity of the bigram model on training data
print('bigram train perplexity:',
      bigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=bigram_model, ngram=2, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

"""### Smoothing

Implement a smoothed version of the bigram model. You may extend the `BigramModel` class and re-use some of the functions.

You will implement the following smoothing techniques:
-  Laplace or add-k smoothing
- Interpolation

**Laplace or Add-k smoothing**
- what is the effect of changing `k`?
"""

"""choose your hyperparameter and see the difference in performance"""

# ADD YOUR CODE HERE

# CHANGE THE None VALUES TO YOUR DESIRED VALUES
# Please feel free to play with these hyperparameters to see the effects on the
# quality of generated names and perplexity

BIGRAM_LAPLACE_K = 0.75 # value of k for add-k or Laplac smoothing in bigram models

"""
Implementation of a bigram model with laplace or add-k smoothing.

"""

class LaplaceSmoothedBigramModel(BigramModel):
    # This class extends BigramModel.

    def __init__(self, train_text, k):
        super().__init__(train_text)
        self.k = k # specify k for smoothing

    # You should override ONLY those functions
    # which calculate probability of a bigram.
    # You can override get_next_char_probabilities
    # or any other helper functions you use in BigramModel
    # to calculate bigram probabilities.

    # ADD YOUR CODE HERE
    # BEGIN CODE
    def get_name_log_probability(self, name):
        log_prob = 0.0
        char_probs = self.get_next_char_probabilities()
        for i in range(len(name) - 1):
            name1 = name[i]
            name2 = name[i + 1]
            if name1 not in char_probs:
                continue
            if name2 not in char_probs[name1]:
                total = sum(char_probs[name1].values())
                log_prob += np.log(self.k/total + self.k*len(self.vocabulary[name1]))
            else:
                log_prob += np.log(char_probs[name1][name2])
        return log_prob
        # END CODE

# smoothed_bigram_model = LaplaceSmoothedBigramModel(train_text, k=0.5)
# print('smoothed bigram train perplexity:',
#       smoothed_bigram_model.get_perplexity(train_text))
# smoothed_bigram_model = LaplaceSmoothedBigramModel(train_text, k=0.7)
# print('smoothed bigram train perplexity:',
#       smoothed_bigram_model.get_perplexity(train_text))
# smoothed_bigram_model = LaplaceSmoothedBigramModel(train_text, k=0.9)
# print('smoothed bigram train perplexity:',
#       smoothed_bigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

smoothed_bigram_model = LaplaceSmoothedBigramModel(train_text, k=BIGRAM_LAPLACE_K)

# check the perplexity of the bigram model on training data
print('smoothed bigram train perplexity:',
      smoothed_bigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=smoothed_bigram_model, ngram=2, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

"""**Interpolation**
- what are good values for `lambdas` in interpolation?
"""

"""choose your hyperparameter and see the difference in performance"""

# ADD YOUR CODE HERE

# CHANGE THE None VALUES TO YOUR DESIRED VALUES
# Please feel free to play with these hyperparameters to see the effects on the
# quality of generated names and perplexity

BIGRAM_LAMBDAS = (0.7, 0.3) # lambdas for interpolation smoothing in bigram models

"""
Implementation of a bigram model with interpolation smoothing
"""

class InterpolationSmoothedBigramModel(BigramModel):

    def __init__(self, train_text, lambdas):
        super().__init__(train_text)
        self.lambda_1, self.lambda_2 = lambdas

    # You should override ONLY those functions
    # which calculate probability of a bigram.
    # You can override get_next_char_probabilities
    # or any other helper functions you use in BigramModel
    # to calculate bigram probabilities.

    # ADD YOUR CODE HERE
    # BEGIN CODE
    def get_name_log_probability(self, name):
        log_prob = 0.0
        char_probs = self.get_next_char_probabilities()
        for i in range(len(name) - 1):
            name1 = name[i]
            name2 = name[i + 1]
            if name1 not in char_probs or name2 not in char_probs[name1]:
                p_name1 = 1 / len(self.vocabulary)
                pn1pn2 = 1 / len(self.vocabulary)
                log_prob += np.log(self.lambda_1 * p_name1 + self.lambda_2 * pn1pn2)
            else:
                log_prob += np.log(char_probs[name1][name2])
        return log_prob
        # END CODE

## Please do not change anything in this code block.

smoothed_bigram_model = InterpolationSmoothedBigramModel(train_text, lambdas=BIGRAM_LAMBDAS)

# check the perplexity of the bigram model on training data
print('smoothed bigram train perplexity:',
      smoothed_bigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=smoothed_bigram_model, ngram=2, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

# Release models we don't need any more.
del bigram_model
del smoothed_bigram_model

"""## 1.3 Trigram (smoothed)"""

"""choose your hyperparameter and see the difference in performance"""

# ADD YOUR CODE HERE

# CHANGE THE None VALUES TO YOUR DESIRED VALUES
# Please feel free to play with these hyperparameters to see the effects on the
# quality of generated names and perplexity

TRIGRAM_LAMBDAS = (0.2, 0.2, 0.6) # lambdas for interpolation smoothing in trigram models
# TRIGRAM_LAMBDAS = (0.1, 0.3, 0.6) # lambdas for interpolation smoothing in trigram models
# TRIGRAM_LAMBDAS = (0.3, 0.2, 0.5) # lambdas for interpolation smoothing in trigram models

class TrigramModel(NGramLanguageModel):
    def __init__(self, train_text):
        """
        Initialize and train the model with train_text.

        Args:
            train_text (list of list): List of tokenized names
        """
        # BEGIN CODE
        self.train_text = train_text
        self.char_counts = get_trigram_counts(train_text)

        self.vocab = get_trigram_counts(train_text)
        total_counts = get_trigram_counts(train_text)
        self.char_probs = get_trigram_counts(train_text)
        self.train_text = train_text
        self.char_counts = get_trigram_counts(train_text)
        vocab =get_trigram_counts(train_text)
        for key, value in vocab.items():
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, dict):
                    vocab[key][inner_key] = set(inner_value.keys())
        self.vocab=vocab
        for outer_key,outer_val  in self.char_counts.items():
            for inner_key,inner_val in self.char_counts[outer_key].items():
                if isinstance(inner_val, dict):
                    total_counts[outer_key][inner_key] = sum(inner_val.values())
        self.total= total_counts

        char_probs = get_trigram_counts(train_text)
        for outer_key,outer_val in char_probs.items():
            for inner_key,inner_val in char_probs[outer_key].items():
                if isinstance(inner_val, dict):
                    for key,value in char_probs[outer_key][inner_key].items():
                        char_probs[outer_key][inner_key][key] = value/self.total[outer_key][inner_key]
        self.char_probs = char_probs
        # END CODE
    def get_next_char_probabilities(self):
        """
        Returns a probability distribution over all chars in the vocabulary.
        Probability distribution should sum to one.
        """
        return self.char_probs

    def get_name_log_probability(self, name):
        """
        Calculates the log probability of name according to the trigram model.

        Args:
            name (list): List of tokens

        Returns:
            Log probability of the name (float)
        """
        # BEGIN CODE
        log_prob = 0.0
        char_probs = self.get_next_char_probabilities()
        for i in range(len(name) - 2):
            char1 = name[i]
            char2 = name[i + 1]
            char3 = name[i + 2]
            if char1 not in char_probs or char2 not in char_probs[char1] or char3 not in char_probs[char1][char2]:
                prob_char1 = 1 / len(self.vocab)
                prob_char2_given_char1 = 1 / len(self.vocab)
                prob_char3_given_char1_char2 = 1 / len(self.vocab)
                log_prob += np.log(0.2* prob_char1 + 0.2 * prob_char2_given_char1 + 0.6 * prob_char3_given_char1_char2)
            else:
                log_prob += np.log(char_probs[char1][char2][char3])

        return log_prob
        # END CODE

    def get_perplexity(self, text):
            """
            Returns the perplexity of the model on a text as a float.

            Args:
                text [list]: a list of string tokens

            Returns:
                perplexity of the given text [float]
            """

            # ADD YOUR CODE HERE
            # BEGIN CODE
            log_prob =0.0
            n=0
            for name in text:
                log_prob += self.get_name_log_probability(name)
                n += len(name)

            perplexity = math.exp(-log_prob/n)
            return perplexity
            # END CODE

    def sample_next_char (self,name, temperature):
        # BEGIN CODE
        prev_char = name[-1]
        prev_prev_char = name[-2]

        temp = self.char_probs.get(prev_prev_char, {})

        next_char_probs = temp.get(prev_char,{})

        if temperature > 0.0:
            probs = [p ** (1 / temperature) for p in next_char_probs.values()]
            sum_probs = sum(probs)
            probs = [p / sum_probs for p in probs]
        else:
            probs = next_char_probs.values()
        # END CODE
        next_char = random.choices(list(next_char_probs.keys()), weights=probs)[0]
        if next_char not in self.vocab[prev_prev_char][prev_char]:
            next_char = random.choice(list(self.vocab[prev_char]))

        return next_char

    def sample_from_bigram(self,name,temperature):
        unigram_model = BigramModel(train_text)
        return unigram_model.sample_next_char(name,temperature)

    def generate_names(self, k, n=MAX_NAME_LENGTH, prefix=None):
        """
        Given a prefix, generate k names according to the model.
        The default prefix is None.

        Args:
            k [int]: Number of names to generate
            n [int]: Maximum length (number of tokens) in the generated name
            prefix [list of tokens]: Prefix after which the names have to be generated

        Returns:
            list of generated names [list]
        """

        # ADD YOUR CODE HERE
        # BEGIN CODE
        names= []
        for _ in range(k):
            if prefix:
                if prefix[-1] == '</s>':
                    prefix = prefix[:-1]
                name=prefix
            else:
                name=['<s>']
                next_c = self.sample_from_bigram(name,0.5)
                name.append(next_c)
            while len(name) <n and name[-1]!='</s>':
                next_char = self.sample_next_char(name,0.5)
                if next_char == "<unk>":
                    next_char = self.sample_from_bigram(name,0.5)
                name.append(next_char)
            names.append(name)
        generated_name = [''.join(sublist) for sublist in names]
        return generated_name
        # END CODE

    def get_most_likely_chars(self, sequence, k):
        """
        Given a sequence of characters, outputs k most likely characters after the sequence.

        Args:
            sequence [list[str]]: list of characters
            k [int]: number of most likely characters to return

        Returns:
            chars [list[str]]: *Ordered* list of most likely characters
                        (with charcater at index 0 being the most likely and
                        character at index k-1 being the least likely)

        """
        # ADD YOUR CODE HERE
        # BEGINC ODE
        last_char = sequence[-1]
        last_last_char = sequence[-2]
        next_char_probs = self.char_probs.get(last_last_char, {}).get(last_char,{})
        valid_chars = self.vocab[last_last_char][last_char]
        next_char_probs = {char: prob for char, prob in next_char_probs.items() if char in valid_chars}
        sorted_chars = sorted(next_char_probs, key=next_char_probs.get, reverse=True)
        most_likely_chars = sorted_chars[:k]
        # END CODE
        return most_likely_chars

"""#### Eval"""

## Please do not change anything in this code block.

trigram_model = TrigramModel(train_text)

print('trigram train perplexity:',
      trigram_model.get_perplexity(train_text))

## Please do not change anything in this code block.

eval_ngram_model(model=trigram_model, ngram=3, ds=validation_text, ds_name='validation', eval_prefixes=eval_prefixes, eval_sequences=eval_sequences, num_names=5)

# Release models we don't need any more.
del trigram_model

"""# Module 2: Neural Language Modelling

## 2.1 Neural N-gram Language Model

For this part of the assignment, you should use the GPU (you can do this by changing the runtime of this notebook).

In this section, you will implement a neural version of an n-gram model.  The model will use a simple feedforward neural network that takes the previous `n-1` chars and outputs a distribution over the next char.

You will use PyTorch to implement the model.  We've provided a little bit of code to help with the data loading using [PyTorch's data loaders](https://pytorch.org/docs/stable/data.html)
"""

# Import the necessary libraries
