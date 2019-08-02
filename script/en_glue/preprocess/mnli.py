import sys

mapping = {
    'contradiction': 0,
    'neutral': 1,
    'entailment': 2,
}

i = 0
for line in sys.stdin:
    arr = line.strip().split('\t')

    if len(arr) == 12:
        if i == 0:
            i += 1
            print('text_a\ttext_b\tlabel')
            continue
        print("{}\t{}\t{}".format(arr[8], arr[9], mapping[arr[11]]))
    elif len(arr) == 16:
        if i == 0:
            i += 1
            print('text_a\ttext_b\tlabel')
            continue
        s1 = arr[8]
        s2 = arr[9]
        s3 = arr[15]
        print("{}\t{}\t{}".format(s1, s2, mapping[s3]))
    else:
        if i == 0:
            i += 1
            print('qid\ttext_a\ttext_b\tlabel')
            continue
        qid = arr[0]
        s1 = arr[8]
        s2 = arr[9]
        print("{}\t{}\t{}\t-1".format(qid, s1, s2))
