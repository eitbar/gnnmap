CUDA_VISIBLE_DEVICES=0 python3 eval.py "./SRC_SUPERVISED_en-de-ENDEFASTTEXT_10k-nosl.txt" "./TAR_SUPERVISED_en-de-ENDEFASTTEXT_10k-nosl.txt" \
                                        -d ./data/en-de/en-de.5000-6500.txt \
                                        --cuda \
                                        --retrieval nn