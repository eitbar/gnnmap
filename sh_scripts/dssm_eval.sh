CUDA_VISIBLE_DEVICES=3 python3 eval_use_csls.py "./data/en-zh/linear.wiki.10k.en-zh-aligned.EN.vec" "./data/en-zh/linear.wiki.10k.en-zh-aligned.ZH.vec" \
                                        -d ./data/en-zh/en-zh.5000-6500.txt \
                                        --retrieval nn \
                                        --cuda \
                                        --model ./data/en-zh/linear.en-zh.ENHR-model.pickle \
                                        --eval_result eval_result_nn_for_iden_graph.json