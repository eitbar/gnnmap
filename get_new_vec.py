import embeddings
import argparse


def main():

    parser = argparse.ArgumentParser(description='Run classification based self learning for aligning embedding spaces in two languages.')

    parser.add_argument('--train_dict', type=str, help='Name of the input dictionary file.', required = True)
    parser.add_argument('--val_dict', type=str, help='Name of the input dictionary file.', required = True)
    parser.add_argument('--in_src', type=str, help='Name of the input source languge embeddings file.', required = True)
    parser.add_argument('--in_tar', type=str, help='Name of the input target language embeddings file.', required = True)
    parser.add_argument('--out_src', type=str, help='Name of the output source languge embeddings file.', required = True)
    parser.add_argument('--out_tar', type=str, help='Name of the output target language embeddings file.', required = True)

    args = parser.parse_args()

    dtype = "float32"
    srcfile = open(args.in_src, encoding="utf-8", errors='surrogateescape')
    trgfile = open(args.in_tar, encoding="utf-8", errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, 200000, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, 200000, dtype=dtype)

    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    src_ind2word = {i: word for i, word in enumerate(src_words)}
    trg_ind2word = {i: word for i, word in enumerate(trg_words)}
    src_word = []
    tgt_word = []
    f = open(args.train_dict, encoding="utf-8", errors='surrogateescape')
    for line in f:
        src, trg = [_.lower().strip() for _ in line.split()]
        src_word.append(src)
        tgt_word.append(trg)
    f = open(args.val_dict, encoding="utf-8", errors='surrogateescape')
    for line in f:
        src, trg = [_.lower().strip() for _ in line.split()]
        src_word.append(src)
        tgt_word.append(trg)
        
    src_word = list(set(src_word))
    tgt_word = list(set(tgt_word))
    src_oov = [w for w in src_word if w not in src_word2ind]
    tgt_oov = [w for w in tgt_word if w not in trg_word2ind]
    print(src_oov)
    print(tgt_oov)


    src_word = [w for w in src_word if w not in src_oov]
    tgt_word = [w for w in tgt_word if w not in tgt_oov]

    src_word_index = [src_word2ind[w] for w in src_word]
    tgt_word_index = [trg_word2ind[w] for w in tgt_word]

    print(len(src_word_index))
    print(len(tgt_word_index))

    i = 0
    while(len(src_word_index) < 30000):
        if i not in src_word_index:
            src_word_index.append(i)
        i += 1

    i = 0
    while(len(tgt_word_index) < 30000):
        if i not in tgt_word_index:
            tgt_word_index.append(i)
        i += 1

    src_word_index = sorted(src_word_index)
    tgt_word_index = sorted(tgt_word_index)

    src_words = [src_ind2word[_] for _ in src_word_index]
    trg_words = [trg_ind2word[_] for _ in tgt_word_index]

    new_x = x[src_word_index]
    new_z = z[tgt_word_index]            


    srcfile = open(args.out_src, mode='w', encoding="utf-8", errors='surrogateescape')
    trgfile = open(args.out_tar, mode='w', encoding="utf-8", errors='surrogateescape')
    embeddings.write(src_words, new_x, srcfile)
    embeddings.write(trg_words, new_z, trgfile)
    srcfile.close()
    trgfile.close()

if __name__ == "__main__":
    main()