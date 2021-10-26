## run

直接用gnn-dssm从候选词（3w）中选择score最高的词作为结果

`bash dssm_train.sh`

or

套用classification based method，用gnn-dssm替代原classifier，用于迭代扩充seed dict时过滤一些噪音

`bash new_train.sh`