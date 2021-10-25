CUDA_VISIBLE_DEVICES=3 python tmp_classymap.py --train_dict "./data/en-hr/yacle.train.freq.3k.en-hr.tsv" \
                                            --in_src "./data/en-hr/wiki.30k.en.vec" \
                                            --in_tar "./data/en-hr/wiki.30k.hr.vec" \
                                            --src_lid "en" \
                                            --tar_lid "hr" \
                                            --out_src "./data/en-hr/class.wiki.30k.en-hr-aligned.EN.vec" \
                                            --out_tar "./data/en-hr/class.wiki.30k.en-hr-aligned.HR.vec" \
                                            --model_filename "./data/en-hr/class.en-hr.ENHR-model.pickle" \
                                            --idstring ENHRFASTTEXT