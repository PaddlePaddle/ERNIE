import sys

ans_dict = {}
text_ans_dict = {}

filename = './data/flickr/flickr.dev.data'
with open(filename) as f:
    for line in f:
        line = line.strip().split('\t')
        image_id, sent_id = line[0], line[1]
        ans_dict[sent_id.strip(' ')] = image_id.strip(' ')
        text_ans_dict.setdefault(image_id.strip(' '), [])
        text_ans_dict[image_id.strip(' ')].append(sent_id.strip(' '))

if len(sys.argv) > 1:
    res_file = sys.argv[1]
else:
    res_file = "./result"
print ('=============== IMAGE RETRIEVAL ==================')
with open(res_file) as f:
    r1, r5, r10 = 0, 0, 0
    cnt = 0
    res_dict = {}
    text_res_dict = {}
    idx_all = 0.0
    for line in f:
        line = line.strip().split('\t')
        if len(line) != 3:
            break
        score, image_id, sent_id = float(line[0]), line[1], line[2]
        res_dict.setdefault(sent_id, [])
        res_dict[sent_id].append((score, image_id))
        text_res_dict.setdefault(image_id, [])
        text_res_dict[image_id].append((score, sent_id))
        if len(res_dict[sent_id]) == 1000:
            res_list = res_dict[sent_id]
            res_list = sorted(res_list, reverse = True)
            ans = ans_dict[sent_id]
            image_id_sort = list(zip(*res_list)[1])
            ans_idx = image_id_sort.index(ans.strip())
            if ans_idx < 1:
                r1 += 1.0
            if ans_idx < 5:
                r5 += 1.0
            if ans_idx < 10:
                r10 += 1.0
            idx_all += (ans_idx + 1)
            cnt += 1
            if cnt %  100 == 0:
                print cnt, round(r1/cnt, 4), round(r5/cnt, 4), round(r10/cnt, 4), round(idx_all/cnt, 4)
    print '-----------------------------'
    print "instance %d r1:%.4f, r5:%.4f, r10:%.4f, avg_rank:%.4f" % (cnt, r1/cnt, r5/cnt, r10/cnt, idx_all/cnt)

print ('\n=============== TEXT RETRIEVAL ==================')
cnt = 0
r1, r5, r10 = 0, 0, 0
idx_all = 0.0
for image_id in text_res_dict:
    res_list = text_res_dict[image_id]
    res_list = sorted(res_list, reverse = True)
    ans = text_ans_dict[image_id]
    text_id_sort = list(zip(*res_list)[1])
    ans_idx_all = []
    for item in ans: 
        ans_idx_all.append(text_id_sort.index(item.strip()))
    ans_idx = min(ans_idx_all)
    if ans_idx < 1:
        r1 += 1.0
    if ans_idx < 5:
        r5 += 1.0
    if ans_idx < 10:
        r10 += 1.0
    idx_all += (ans_idx + 1)
    cnt += 1
    if cnt % 500 == 0:
        print cnt, round(r1/cnt, 4), round(r5/cnt, 4), round(r10/cnt, 4), round(idx_all/cnt, 4)

print '-----------------------------'
print "instance %d r1:%.4f, r5:%.4f, r10:%.4f, avg_rank:%.4f" % (cnt, r1/cnt, r5/cnt, r10/cnt, idx_all/cnt)
