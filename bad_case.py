import json

def main():
    with open('eval_result_csls_for_iden_graph_train.json', 'r') as f:
        dssm_csls_result = json.load(f)
    with open('eval_result_nn_for_iden_graph_train.json', 'r') as f:
        dssm_nn_result = json.load(f)

    with open('debug_for_vecmap_csls_train.json', 'r') as f:
        vecmap_csls_result = json.load(f)

    with open('debug_for_vecmap_nn_train.json', 'r') as f:
        vecmap_nn_result = json.load(f)

    with open('src_word2nn.json', 'r') as f:
        src_word2nn = json.load(f)

    with open('neg_select.json', 'r') as f:
        neg_select_list = json.load(f)
        neg_select = {}
        for d in neg_select_list:
            neg_select[d['src_word']] = d['hard_neg_sample_words']



    train_srcs = dssm_csls_result.keys()

    result_p1 = []
    result_p5 = []
    result_p10 = []
    result_o = []
    for src in train_srcs:
        src_str = src
        gold_str = ', '.join(dssm_csls_result[src]['gold'])
        dssm_csls_pred = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in dssm_csls_result[src]['pred']][:50])
        vecmap_csls_pred = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in vecmap_csls_result[src]['pred']][:50])
        dssm_nn_pred = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in dssm_nn_result[src]['pred']][:50])
        vecmap_nn_pred = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in vecmap_nn_result[src]['pred']][:50])
        neg_example_select = neg_select[src_str]
        monolingual_nn = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in src_word2nn[src]][:10])

        item = {
            'src_word': src,
            'gold_word': gold_str,
            'monolingual_nn': monolingual_nn,
            'hard_neg_example_select': neg_example_select,
            'dssm_csls_pred': dssm_csls_pred,
            'vecmap_csls_pred': vecmap_csls_pred,
            'dssm_nn_pred': dssm_nn_pred,
            'vecmap_nn_pred': vecmap_nn_pred
        }

        dssm_p1 = [dssm_csls_result[src]['pred'][0][0]]
        dssm_p5 = [dssm_csls_result[src]['pred'][i][0] for i in range(5)]
        dssm_p10 = [dssm_csls_result[src]['pred'][i][0] for i in range(10)]

        if len(set(dssm_csls_result[src]['gold']) & set(dssm_p1)) > 0:
            result_p1.append(item)
        elif len(set(dssm_csls_result[src]['gold']) & set(dssm_p5)) > 0:
            result_p5.append(item)
        elif len(set(dssm_csls_result[src]['gold']) & set(dssm_p10)) > 0:
            result_p10.append(item)
        else:
            result_o.append(item)

    result = {
        "p1": result_p1,
        "p5": result_p5,
        "p10": result_p10,
        "o": result_o
    }
    
    with open('bad_case_train.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def test():
    with open('eval_result_csls_for_iden_graph.json', 'r') as f:
        dssm_csls_result = json.load(f)
    with open('eval_result_nn_for_iden_graph.json', 'r') as f:
        dssm_nn_result = json.load(f)

    with open('debug_for_vecmap_csls.json', 'r') as f:
        vecmap_csls_result = json.load(f)

    with open('debug_for_vecmap_nn.json', 'r') as f:
        vecmap_nn_result = json.load(f)

    with open('src_word2nn.json', 'r') as f:
        src_word2nn = json.load(f)

    test_srcs = dssm_csls_result.keys()

    result_p1 = []
    result_p5 = []
    result_p10 = []
    result_o = []
    for src in test_srcs:
        src_str = src
        gold_str = ', '.join(dssm_csls_result[src]['gold'])
        dssm_csls_pred = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in dssm_csls_result[src]['pred']][:50])
        vecmap_csls_pred = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in vecmap_csls_result[src]['pred']][:50])
        dssm_nn_pred = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in dssm_nn_result[src]['pred']][:50])
        vecmap_nn_pred = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in vecmap_nn_result[src]['pred']][:50])
        monolingual_nn = ', '.join([w + '(' + "%.5f" % s + ')' for w, s in src_word2nn[src]][:10])

        item = {
            'src_word': src,
            'gold_word': gold_str,
            'monolingual_nn': monolingual_nn,
            'dssm_csls_pred': dssm_csls_pred,
            'vecmap_csls_pred': vecmap_csls_pred,
            'dssm_nn_pred': dssm_nn_pred,
            'vecmap_nn_pred': vecmap_nn_pred
        }

        dssm_p1 = [dssm_csls_result[src]['pred'][0][0]]
        dssm_p5 = [dssm_csls_result[src]['pred'][i][0] for i in range(5)]
        dssm_p10 = [dssm_csls_result[src]['pred'][i][0] for i in range(10)]

        if len(set(dssm_csls_result[src]['gold']) & set(dssm_p1)) > 0:
            result_p1.append(item)
        elif len(set(dssm_csls_result[src]['gold']) & set(dssm_p5)) > 0:
            result_p5.append(item)
        elif len(set(dssm_csls_result[src]['gold']) & set(dssm_p10)) > 0:
            result_p10.append(item)
        else:
            result_o.append(item)

    result = {
        "p1": result_p1,
        "p5": result_p5,
        "p10": result_p10,
        "o": result_o
    }
    
    with open('bad_case_test.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)    

if __name__ == "__main__":
    main()
    test()