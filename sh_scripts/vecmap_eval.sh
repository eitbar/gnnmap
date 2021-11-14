CUDA_VISIBLE_DEVICES=0 python3 eval.py "./SRC_SUPERVISED_en-zh-ENZHFASTTEXT_10k-nosl.txt" "./TAR_SUPERVISED_en-zh-ENZHFASTTEXT_10k-nosl.txt" \
                                        -d ./data/en-zh/en-zh.5000-6500.txt \
                                        --cuda \
                                        --retrieval csls