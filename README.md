## work dir

主要文件

```
|
|-- dssm_train.py  # dssm 训练主函数
|-- new_dssm.py    # dssm class 代码
|-- DenseGraphConv.py  # gnn class 代码
|-- art_wrapper.py     # 借助vecmap获取初始embedding的相关代码
|-- eval_use_csls.py   # evaluate 代码
|-- gnn.yaml           # conda 环境
|-- sh_scripts         # 一些运行脚本
|  |-- ...
|-- ...
```

## run

直接用gnn-dssm从候选词（3w）中选择score最高的词作为结果

`bash sh_scripts/dssm_train.sh`

原本vecmap的基础上，添加图结构降噪

`bash sh_scripts/vecmap_train.sh`

套用classification based method，用gnn-dssm替代原classifier，用于迭代扩充seed dict时过滤一些噪音

`bash sh_scripts/new_train.sh`