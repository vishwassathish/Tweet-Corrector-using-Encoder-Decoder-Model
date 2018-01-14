# Tweet-Corrector-using-Encoder-Decoder-Model
1. INPUT

->shape of input vector (20, 100) <br />
->each word is represented as a vector of 100 features <br />
->sentences with more than 20 words are clipped <br />
->and one with less than 20 words are padded with zero vectors <br />
->target : one hot vectors of dimension = length of vocabulary <br />

2. OUTPUT

->softmax output 

Approach Taken :-

1. First we load all the tweet data from the file 'consolidate.csv' and separately store original and corrected 
tweets in two lists after tokenizing them. We used nltk library to tokenize sentences into a list of words.

2. The original data is then preprocessed. Each word is converted to its lowercase words, to maintain uniformity 
in the dataset. The corrected tweets are processed to find all the unique words and their count. Only a subset of 
these unique words are chosen according to their occurrence count for our bag of words.

3. We Created one-hot vectors for each word in our bag of words, which will constitute our 'expected output' data.
Each word in the original tweet dataset is converted to its corresponding vector. Gensim's word2vec was used for 
this purpose.

4. Now that we had all the required data in their proper format, we segregated and randomly chose X(input) and 
y(expected output) vectors from the dataset. This data was split into training(4050) and validation data(50). 

5. For our network :- 
Model used : encoder-decoder RNN model using keras
error function : Categorical cross entropy 
activation : softmax 
