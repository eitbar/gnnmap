CUDA_VISIBLE_DEVICES=1 python vecmap_with_graph.py --train_dict "./data/en-de/en-de.0-5000.txt" \
                                            --in_src "./data/en-de/wiki.10k.en.vec" \
                                            --in_tar "./data/en-de/wiki.10k.de.vec" \
                                            --src_lid "en" \
                                            --tar_lid "de" \
                                            --idstring ENDEFASTTEXT_10k