import sys

mapping = {'entailment': 1, 'not_entailment': 0}

i = 0
for line in sys.stdin:
    arr = line.strip().split('\t')
    s1 = arr[1]
    s2 = arr[2]
    if len(arr) == 4:
        if i == 0:
            i += 1
            print('text_a\ttext_b\tlabel')
            continue
        s3 = arr[3]
        print("{}\t{}\t{}".format(s1, s2, mapping[s3]))
    else:
        if i == 0:
            i += 1
            print('qid\ttext_a\ttext_b\tlabel')
            continue
        print("{}\t{}\t{}\t-1".format(arr[0], s1, s2))
