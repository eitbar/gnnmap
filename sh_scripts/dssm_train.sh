CUDA_VISIBLE_DEVICES=1 python dssm_train.py --train_dict "./data/en-zh/en-zh.0-5000.txt" \
                                            --val_dict "./data/en-zh/en-zh.5000-6500.txt" \
                                            --in_src "./data/en-zh/wiki.10k.en.vec" \
                                            --in_tar "./data/en-zh/wiki.10k.zh.vec" \
                                            --out_src "./data/en-zh/en.whitening.tmp.vec" \
                                            --out_tar "./data/en-zh/zh.whitening.tmp.vec" \
                                            --model_filename "./data/en-zh/ENZH-model.tmp.pickle" \
                                            --train_batch_size 256 \
                                            --train_epochs 200 \
                                            --use_whitening "post" \
                                            --whitening_data "train" \
                                            --random_neg_per_pos 256 \
                                            --hard_neg_per_pos 256 \
                                            --hard_neg_sampling_method "ab" \
                                            --hard_neg_top_k 500 \
                                            --hard_neg_random \
                                            --h_dim 300 \
                                            --lr 0.001