import numpy as np

########################################
## index word vectors
########################################

def get_glove_embeddings(glove_filepath, embed_dims):
    print('Indexing glove word vectors')

    embeddings_index = dict()
    with open(glove_filepath, encoding='utf-8') as f:
        for line in f:
            values = line.replace('\xa0','').split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            embed_dims = len(coefs)
            if embed_dims == 300:
                embeddings_index[word] = coefs
            else:
                print ("{} is not {} dims, only {} dims".format(line[:10], embed_dims, len(coefs)))

    vocab_size = len(embeddings_index)
    max_len = max(len(w) for w in embeddings_index)

    print ('Loaded {} word vectors from {}'.format(vocab_size, glove_filepath))
    print ('')
    return embeddings_index, vocab_size, max_len


def get_embedding_matrix(embeddings_index, word_index, max_vocab, embed_dim):
    ########################################
    ## prepare embeddings
    ########################################
    print('Preparing embedding matrix')
    nb_words = min(max_vocab, len(word_index))

    # initialize embedding matrix with zeros
    embedding_matrix = np.zeros((nb_words, embed_dim))

    missing_words = []
    for word, i in word_index.items():
        if i >= max_vocab:
            continue
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            missing_words.append(word)

    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    return embedding_matrix, nb_words, missing_words
