CUDA_VISIBLE_DEVICES=3 python3 old_eval.py "./data/en-hr/noc.wiki.30k.en-hr-aligned.EN.vec" "./data/en-hr/noc.wiki.30k.en-hr-aligned.HR.vec" -d ./data/en-hr/yacle.test.freq.2k.en-hr.tsv --retrieval csls --cuda --model ./data/en-hr/class.en-hr.ENHR-model.pickle