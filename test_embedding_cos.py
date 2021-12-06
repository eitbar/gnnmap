
import embeddings
import argparse


def main(args):
  dtype = "float32"
  #srcfile = open(args.in_src, encoding="utf-8", errors='surrogateescape')
  trgfile = open(args.in_tar, encoding="utf-8", errors='surrogateescape')
  #src_words, x = embeddings.read(srcfile, 10000, dtype=dtype)
  trg_words, z = embeddings.read(trgfile, 10000, dtype=dtype)

  # load the supervised dictionary
  #src_word2ind = {word: i for i, word in enumerate(src_words)}
  trg_word2ind = {word: i for i, word in enumerate(trg_words)}
  #src_ind2word = {i: word for i, word in enumerate(src_words)}
  #trg_ind2word = {i: word for i, word in enumerate(trg_words)}

 
  #embeddings.normalize(x, ['unit', 'center', 'unit'])
  embeddings.normalize(z, ['unit', 'center', 'unit'])

  #src_word_ind = src_word2ind[args.src_word]
  #trg_word_ind = trg_word2ind[args.tgt_word]
  t1_word_ind = trg_word2ind[args.t1]
  t2_word_ind = trg_word2ind[args.t2]

  score = z[t1_word_ind].dot(z[t2_word_ind])

  print(score)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Run classification based self learning for aligning embedding spaces in two languages.')
  parser.add_argument('--in_src', type=str, default='./data/en-zh/wiki.10k.en.vec')
  parser.add_argument('--in_tar', type=str, default='./data/en-zh/wiki.10k.zh.vec')
  parser.add_argument('--t1', type=str, help='Name of the input target language embeddings file.', required = True)
  parser.add_argument('--t2', type=str, help='Name of the input target language embeddings file.', required = True)

  args = parser.parse_args()
  main(args)