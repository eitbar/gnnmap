CUDA_VISIBLE_DEVICES=0 python classymap.py --train_dict "./data/en-hr/yacle.train.freq.5k.en-hr.tsv" \
                                            --val_dict "./data/en-hr/yacle.test.freq.2k.en-hr.tsv" \
                                            --in_src "./data/en-hr/wiki.10k.en.vec" \
                                            --in_tar "./data/en-hr/wiki.10k.hr.vec" \
                                            --src_lid "en" \
                                            --tar_lid "hr" \
                                            --out_src "./data/en-hr/ref.gnn.wiki.10k.en-hr-aligned.EN.vec" \
                                            --out_tar "./data/en-hr/ref.gnn.wiki.10k.en-hr-aligned.HR.vec" \
                                            --model_filename "./data/en-hr/ref.gnn.en-hr.ENHR-model.pickle" \
                                            --idstring ENHRFASTTEXT_10K