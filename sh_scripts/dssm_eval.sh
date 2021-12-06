CUDA_VISIBLE_DEVICES=1 python3 eval_use_csls.py "./data/en-de/wiki.30k.en.vec" \
                                                "./data/en-de/wiki.30k.de.vec" \
                                        -d ./data/en-de/en-de.5000-6500.txt \
                                        --retrieval nn \
                                        --cuda \
                                        --model ./data/en-de/hh.ENDE-model.n5abcw4r.pickle \
                                        --eval_result eval_result_nn_for_iden_graph.json \
                                        --use_origin 