from sentivi.corpus import Corpus

if __name__ == '__main__':
    corpus = Corpus(source_file='./data/dev.vi', n_grams=3)
    # corpus.export_vocab('./data/vocab.vi')
    for x, y in corpus:
        print(x, y)
