CUDA_VISIBLE_DEVICES=1 python vecmap_with_graph.py --train_dict "./data/en-zh/en-zh.0-5000.txt" \
                                            --in_src "./data/en-zh/wiki.10k.en.vec" \
                                            --in_tar "./data/en-zh/wiki.10k.zh.vec" \
                                            --src_lid "en" \
                                            --tar_lid "zh" \
                                            --idstring ENZHFASTTEXT_10k