CUDA_VISIBLE_DEVICES=2 python3 eval_use_csls.py "./data/en-zh/wiki.10k.en.vec" \
                                                "./data/en-zh/wiki.10k.zh.vec" \
                                        -d ./data/en-zh/en-zh.5000-6500.txt \
                                        --retrieval nn \
                                        --cuda \
                                        --model ./data/en-zh/ENZH-model.tmp.pickle \
                                        --eval_result eval_result_nn_for_iden_graph.json \
                                        --use_origin 