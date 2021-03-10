import pandas as pd 
import numpy as np
import pickle
from sklearn.metrics.pairwise import rbf_kernel as _rbf_kernel

from neural.data_loading import get_word_embedder
from preprocessing.utils import denoise_text

from typing import List 

array = np.array


# aggregates features from all word vectors in paragraph by average pooling
def bag_of_embeddings(sents: List[array]) -> array:
    return array([sent.mean(axis=0) for sent in sents])


# computes cosine similarity of each paragraph vector with all other
# from formula: cosine(x, y) = x@y / |x|*|y|
def cosine_similarity(X: array) -> array:
    norms = np.linalg.norm(X, axis=1)
    norms[norms==0] = 1e-08  # avoid division by zero
    X_norm = X / np.expand_dims(norms, axis=1)
    return np.triu(X_norm @X_norm.T) # keep only upper triangular as X symmetric


# applies an RBF kernel over the input paragraph vectors
# RBF(x, y) = var * exp(-gamma * |x - y|^2)
def rbf_kernel(X: array, gamma: float=0.5, var: float=1.) -> array:
    gramm =  var * _rbf_kernel(X, gamma=gamma)
    return np.triu(gramm) # keep only upper triangular as X symmetric


def main(path_to_csv_file: str,
         embeddings: str,
         checkpoint: bool,
         similarity_fn: str,
         similarity_thresh: float
         ):
    
    # parse path and csv file name
    prefix = '/'.join(path_to_csv_file.split('/')[:-1])
    csv_name = path_to_csv_file.split('/')[-1].split('.')[0]

    # use pre-trained word vectors and aggregate features to get paragraph-lvl embeddings
    text, _ = zip(*pd.read_csv(path_to_csv_file, sep=',', header=0).values.tolist())
    text = denoise_text(text)    
    # skip the data vectorization if already run once
    if not checkpoint:
        print('Loading embeddings...')
        embedder = get_word_embedder(embeddings)
        text_embedds = list(map(embedder, text))
        pickle.dump(text_embedds, open(prefix + '/vectors_' + embeddings + '_' + 'similarity.p', 'wb'))
    else:
        text_embedds = pickle.load(open(prefix + '/vectors_' + embeddings + '_' + 'similarity.p', 'rb'))

    text_embedds = [sent.detach().numpy() for sent in text_embedds]
    bags = bag_of_embeddings(text_embedds)

    # define similarity measure (either cosine distance or an RBF kernel)
    if similarity_fn not in ['cosine', 'rbf']:
        raise ValueError('Please select either "cosine" or "rbf" at --similarity_fn')
    sim_fn = rbf_kernel if similarity_fn=='rbf' else cosine_similarity

    print('Computing similarities...')
    similarities = sim_fn(bags) # compute similarities
    similarities[similarities == 1] = 0 # remove main diagonal 
    sim_par_idces = np.argwhere(similarities > similarity_thresh) # get the indices of similar paragraphs

    write_file = f'{embeddings}_{similarity_fn}{similarity_thresh}_similarity_results.txt'
    print(f'Copying most similar paragraphs to {write_file}')
    with open(write_file, 'w+') as f:
        for (i, j) in sim_par_idces:
            f.write(text[i])
            f.write('\n' * 2)
            f.write(text[j])
            f.write('\n' * 3)
            f.write('=' * 256)
            f.write('\n' * 3)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_csv_file', help='path to CSV file containing the data', type=str)
    parser.add_argument('-emb', '--embeddings', help='what type of embedder to use for data vectorization (see code)', type=str, default='glove_lg')
    parser.add_argument('--checkpoint', action='store_true', help='whether to skip dataloading', default=False)
    parser.add_argument('-sim_fn', '--similarity_fn', help='what type of similarity metric to use (cosine distance or RBF)', type=str, default='rbf')
    parser.add_argument('-sim_thr', '--similarity_thresh', help='low similarity bound for grouping paragraphs', type=float, default=0.75)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)