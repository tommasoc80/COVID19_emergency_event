from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

class Embeddings(TransformerMixin):
    '''Transformer object turning a sentence (or tweet) into a single embedding vector'''

    def __init__(self, word_embeds, pool='average'):
        '''
        Required input: word embeddings stored in dict structure available for look-up
        pool: sentence embeddings to be obtained either via average pooling ('average') or max pooing ('max') from word embeddings. Default is average pooling.
        '''
        self.word_embeds = word_embeds
        self.pool_method = pool

    def transform(self, X, **transform_params):
        '''
        Transformation function: X is list of sentence/tweet - strings in the train data. Returns list of embeddings, each embedding representing one tweet
        '''
        return [self.get_sent_embedding(sent, self.word_embeds, self.pool_method) for sent in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_sent_embedding(self, sentence, word_embeds, pool):
        '''
        Obtains sentence embedding representing a whole sentence / tweet
        '''
        # simply get dim of embeddings
        l_vector = len(word_embeds['must'])

        # replace each word in sentence with its embedding representation via look up in the embedding dict strcuture
        # if no word_embedding available for a word, just ignore the word
        # [[0.234234,-0.276583...][0.2343, -0.7356354, 0.123 ...][0.2344356, 0.12477...]...]

        #print(sentence)
        list_of_embeddings = [word_embeds[word.lower()] for word in sentence.split() if word.lower() in word_embeds]

        # Modifica Tommaso - 20201229
        #list_of_embeddings = [word_embeds[token.lower()] for song in sentence_tokenized for token in song if token.lower() in word_embeds]
	    # Obtain sentence embeddings either by average or max pooling on word embeddings of the sentence
        # Option via argument 'pool'
        if pool == 'average':
            sent_embedding = [sum(col) / float(len(col)) for col in zip(*list_of_embeddings)]  # average pooling
        elif pool == 'max':
            sent_embedding = [max(col) for col in zip(*list_of_embeddings)]	# max pooling
        else:
            raise ValueError('Unknown pooling method!')

        # Below case should technically not occur
        if len(sent_embedding) != l_vector:
            sent_embedding = [0] * l_vector

        return sent_embedding
