import argparse
from dssm_train import run_dssm_trainning, NEG_SAMPLING_METHOD
import sys

LR = 0.002
WHITE_SORT = False
SHUFFLE_IN_TRIAN = True
EPOCHES = 200

parser = argparse.ArgumentParser(description='Run classification based self learning for aligning embedding spaces in two languages.')

parser.add_argument('--train_dict', type=str, help='Name of the input dictionary file.')
parser.add_argument('--val_dict', type=str, help='Name of the input dictionary file.')
parser.add_argument('--in_src', type=str, help='Name of the input source languge embeddings file.')
parser.add_argument('--in_tar', type=str, help='Name of the input target language embeddings file.')
parser.add_argument('--out_src', type=str, help='Name of the output source languge embeddings file.')
parser.add_argument('--out_tar', type=str, help='Name of the output target language embeddings file.')
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
parser.add_argument('--update_neg_every_epoch', type=int, default=1, help='recalculate hard neg examples. 0 means fixed hard exmaples.')
parser.add_argument('--random_warmup_epoches', type=int, default=0, help='only use random neg sampling at begin epoches')



parser.add_argument('--train_batch_size', type=int, default=256, help='train batch size')
parser.add_argument('--train_epochs', type=int, default=70, help='train epochs')
parser.add_argument('--eval_every_epoch', type=int, default=5, help='eval epochs')
parser.add_argument('--shuffle_in_train', action='store_true', help='use shuffle in train')
parser.add_argument('--loss_metric', type=str, default="cos", help='number of topk examples for select hard neg word')



parser.add_argument('--model_filename', type=str, help='Name of file where the model will be stored..')
parser.add_argument('--debug', action='store_true', help='store debug info')
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')


parser.add_argument('--adapter_src_cluster_k', type=int, default=5, help='hidden states dim in GNN')
parser.add_argument('--adapter_src_cluster_threshold', type=float, default=0.8, help='learning rate')
parser.add_argument('--adapter_tgt_cluster_k', type=int, default=5, help='hidden states dim in GNN')
parser.add_argument('--adapter_tgt_cluster_threshold', type=float, default=0.8, help='learning rate')
parser.add_argument('--adapter_actfunc', type=str, default="sigmoid", help='learning rate')
parser.add_argument('--adapter_norm', action='store_true', help='use single tower')
parser.add_argument('--adapter_regular', type=float, default=0.005, help='learning rate')
parser.add_argument('--adapter_regular_method', type=str, default="para", help='learning rate')

args = parser.parse_args()

# fixed paras
args.train_dict = "./data/en-zh/en-zh.0-5000.txt"
args.val_dict = "./data/en-zh/en-zh.5000-6500.txt" 
args.in_src = "./data/en-zh/wiki.10k.en.vec" 
args.in_tar = "./data/en-zh/wiki.10k.zh.vec" 
args.out_src = "./data/en-zh/en.whitening.tmp.vec"
args.out_tar = "./data/en-zh/zh.whitening.tmp.vec"
args.model_filename = "./data/en-zh/ENZH-model.tmp.pickle"
# TODO: add early-stop
#args.lr = trial.suggest_categorical(name="lr", choices=[0.0001, 0.0005, 0.001, 0.002])
#args.whitening_sort = trial.suggest_categorical(name="whitening_sort", choices=[True, False])
#args.shuffle_in_train = trial.suggest_categorical(name="shuffle_in_train", choices=[True, False])

# For multi gpu matual para
args.lr = LR
args.whitening_sort = WHITE_SORT
args.shuffle_in_train = SHUFFLE_IN_TRIAN
args.train_epochs = EPOCHES

args.eval_every_epoch = 1
args.hard_neg_random = True
args.hard_sim_method = "cos"


args.train_batch_size = 256
args.loss_metric = "csls"

args.use_whitening = None
args.whitening_data = "train"

args.hard_neg_per_pos = 512
args.hard_neg_top_k = 500
args.random_neg_per_pos = 128
args.hard_neg_random_with_prob = True

args.h_dim = 300

#args.hard_neg_per_pos = trial.suggest_int(name="hard_neg_per_pos", low=64, high=256, step=64)
args.hard_neg_sampling_method = "a"

#args.hard_neg_sampling_threshold = trial.suggest_uniform("hard_neg_sampling_threshold", -0.9, 0.9)
args.random_seed = 7777967

args.update_neg_every_epoch = 1
args.random_warmup_epoches = 1

run_dssm_trainning(args, is_optuna=False)




