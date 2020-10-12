import sys
import json

name = 'book'
if len(sys.argv) > 1:
    name = sys.argv[1]

item_cate = {}
item_map = {}
cate_map = {}
with open('./data/%s_data/%s_item_map.txt' % (name, name), 'r') as f:
    for line in f:
        conts = line.strip().split(',')
        item_map[conts[0]] = conts[1]

if name == 'taobao':
    with open('UserBehavior.csv', 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            iid = conts[1]
            if conts[3] != 'pv':
                continue
            cid = conts[2]
            if iid in item_map:
                if cid not in cate_map:
                    cate_map[cid] = len(cate_map) + 1
                item_cate[item_map[iid]] = cate_map[cid]
elif name == 'book':
    with open('meta_Books.json', 'r') as f:
        for line in f:
            r = eval(line.strip())
            iid = r['asin']
            cates = r['categories']
            if iid not in item_map:
                continue
            cate = cates[0][-1]
            if cate not in cate_map:
                cate_map[cate] = len(cate_map) + 1
            item_cate[item_map[iid]] = cate_map[cate]

with open('./data/%s_data/%s_cate_map.txt' % (name, name), 'w') as f:
    for key, value in cate_map.items():
        f.write('%s,%s\n' % (key, value))
with open('./data/%s_data/%s_item_cate.txt' % (name, name), 'w') as f:
    for key, value in item_cate.items():
        f.write('%s,%s\n' % (key, value))
