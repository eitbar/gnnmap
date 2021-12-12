import optuna
import argparse
from dssm_train import run_dssm_trainning


def objective(trial):

  parser = argparse.ArgumentParser(description='Run classification based self learning for aligning embedding spaces in two languages.')

  parser.add_argument('--train_dict', type=str, help='Name of the input dictionary file.', required = True)
  parser.add_argument('--val_dict', type=str, help='Name of the input dictionary file.', required = True)
  parser.add_argument('--in_src', type=str, help='Name of the input source languge embeddings file.', required = True)
  parser.add_argument('--in_tar', type=str, help='Name of the input target language embeddings file.', required = True)
  parser.add_argument('--out_src', type=str, help='Name of the output source languge embeddings file.', required = True)
  parser.add_argument('--out_tar', type=str, help='Name of the output target language embeddings file.', required = True)
  # parameters

  parser.add_argument('--use_whitening', type=str, choices=["pre", "post", None], default=None, help='use whitening transformation before neg sampling as preprocess') 
  parser.add_argument('--whitening_data', type=str, choices=["train", "all"], default="train", help='use whitening transformation before neg sampling as preprocess') 
  parser.add_argument('--whitening_sort', action='store_true', help='sort whitening data')

  # model related para
  parser.add_argument('--is_single_tower', action='store_true', help='use single tower')
  parser.add_argument('--h_dim', type=int, default=300, help='hidden states dim in GNN')
  parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

  parser.add_argument('--hard_neg_per_pos', type=int, default=256, help='number of hard negative examples')
  parser.add_argument('--hard_neg_sampling_method', type=str, choices=NEG_SAMPLING_METHOD.keys(), default='ab', help='method of neg sampling')
  parser.add_argument('--hard_neg_sampling_threshold', type=float, default=-10, help='filter low similarity neg word in sampling hard word')
  parser.add_argument('--hard_neg_random', action='store_true', help='random sampling hard in every epoch')
  parser.add_argument('--hard_neg_top_k', type=int, default=500, help='number of topk examples for select hard neg word')
  parser.add_argument('--hard_sim_method', type=str, default="cos", help='number of topk examples for select hard neg word')
  parser.add_argument('--hard_neg_random_with_prob', action='store_true', help='random sampling hard in every epoch')
  parser.add_argument('--random_neg_per_pos', type=int, default=256, help='number of random negative examples')

  
  
  parser.add_argument('--train_batch_size', type=int, default=256, help='train batch size')
  parser.add_argument('--train_epochs', type=int, default=70, help='train epochs')
  parser.add_argument('--eval_every_epoch', type=int, default=5, help='eval epochs')
  parser.add_argument('--shuffle_in_train', action='store_true', help='use shuffle in train')
  parser.add_argument('--loss_metric', type=str, default="cos", help='number of topk examples for select hard neg word')


  
  parser.add_argument('--model_filename', type=str, help='Name of file where the model will be stored..', required = True)
  parser.add_argument('--debug', action='store_true', help='store debug info')

  args = parser.parse_args()

  args.train_dict = "./data/en-zh/en-zh.0-5000.txt"
  args.val_dict = "./data/en-zh/en-zh.5000-6500.txt" 
  args.in_src = "./data/en-zh/wiki.10k.en.vec" 
  args.in_tgt = "./data/en-zh/wiki.10k.zh.vec" 
  args.out_src = "./data/en-zh/en.whitening.tmp.vec"
  args.out_tar = "./data/en-zh/zh.whitening.tmp.vec"
  args.model_filename = "./data/en-zh/ENZH-model.tmp.pickle"

  args.train_batch_size = trial.suggest_int(name="train_batch_size", low=128, high=512, step=64)
  args.train_epochs = trial.suggest_int(name="train_epochs", low=100, high=300, step=50)
  args.loss_metric = trial.suggest_categorical(name="loss_metric", choices=["cos", "csls"])
  args.whitening_sort = trial.suggest_categorical(name="whitening_sort", choices=[True, False])
  args.shuffle_in_train = trial.suggest_categorical(name="shuffle_in_train", choices=[True, False])
  args.eval_every_epoch = 1

  args.use_whitening = trial.suggest_categorical(name="use_whitening", choices=["pre", "post", None])
  args.whitening_data = trial.suggest_categorical(name="whitening_data", choices=["train", "all"])
  
  args.hard_neg_top_k = trial.suggest_int(name="hard_neg_top_k", low=100, high=1000, step=100)
  args.random_neg_per_pos = trial.suggest_int(name="random_neg_per_pos", low=64, high=args.hard_neg_top_k, step=64)
  args.hard_neg_random = trial.suggest_categorical(name="hard_neg_random", choices=[True, False])
  args.hard_neg_random_with_prob = trial.suggest_categorical(name="hard_neg_random_with_prob", choices=[True, False])
  args.hard_sim_method = trial.suggest_categorical(name="hard_sim_method", choices=["cos", "csls"])

  args.h_dim = trial.suggest_int(name="h_dim", low=150, high=400, step=50)
  args.lr = trial.suggest_categorical(name="lr", choices=[0.0001, 0.0003, 0.0007, 0.001, 0.002, 0.003])
  
  args.hard_neg_sampling_method = trial.suggest_categorical(name="hard_neg_sampling_method", 
                                        choices=["a", "samplewise_a", "ab", "samplewise_ab", "bi_samplewise_a", "bi_samplewise_ab"])

  args.hard_neg_sampling_threshold = trial.suggest_uniform("hard_neg_sampling_threshold", -1, 1)
  
  
  score = run_dssm_trainning(args, is_optuna=True)
  
  return score[0]


study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), 
                            direction="maximize",
                            study_name='example', 
                            storage='sqlite:///example.db')

study.optimize(objective, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))



