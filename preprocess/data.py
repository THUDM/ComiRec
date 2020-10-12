import os
import sys
import json
import random
from collections import defaultdict

random.seed(1230)

name = 'book'
filter_size = 5
if len(sys.argv) > 1:
    name = sys.argv[1]
if len(sys.argv) > 2:
    filter_size = int(sys.argv[2])

users = defaultdict(list)
item_count = defaultdict(int)

def read_from_amazon(source):
    with open(source, 'r') as f:
        for line in f:
            r = json.loads(line.strip())
            uid = r['reviewerID']
            iid = r['asin']
            item_count[iid] += 1
            ts = float(r['unixReviewTime'])
            users[uid].append((iid, ts))

def read_from_taobao(source):
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            uid = int(conts[0])
            iid = int(conts[1])
            if conts[3] != 'pv':
                continue
            item_count[iid] += 1
            ts = int(conts[4])
            users[uid].append((iid, ts))


if name == 'book':
    read_from_amazon('reviews_Books_5.json')
elif name == 'taobao':
    read_from_taobao('UserBehavior.csv')

items = list(item_count.items())
items.sort(key=lambda x:x[1], reverse=True)

item_total = 0
for index, (iid, num) in enumerate(items):
    if num >= filter_size:
        item_total = index + 1
    else:
        break

item_map = dict(zip([items[i][0] for i in range(item_total)], list(range(1, item_total+1))))

user_ids = list(users.keys())
filter_user_ids = []
for user in user_ids:
    item_list = users[user]
    index = 0
    for item, timestamp in item_list:
        if item in item_map:
            index += 1
    if index >= filter_size:
        filter_user_ids.append(user)
user_ids = filter_user_ids

random.shuffle(user_ids)
num_users = len(user_ids)
user_map = dict(zip(user_ids, list(range(num_users))))
split_1 = int(num_users * 0.8)
split_2 = int(num_users * 0.9)
train_users = user_ids[:split_1]
valid_users = user_ids[split_1:split_2]
test_users = user_ids[split_2:]

def export_map(name, map_dict):
    with open(name, 'w') as f:
        for key, value in map_dict.items():
            f.write('%s,%d\n' % (key, value))

def export_data(name, user_list):
    total_data = 0
    with open(name, 'w') as f:
        for user in user_list:
            if user not in user_map:
                continue
            item_list = users[user]
            item_list.sort(key=lambda x:x[1])
            index = 0
            for item, timestamp in item_list:
                if item in item_map:
                    f.write('%d,%d,%d\n' % (user_map[user], item_map[item], index))
                    index += 1
                    total_data += 1
    return total_data

path = './data/' + name + '_data/'
if not os.path.exists(path):
    os.mkdir(path)

export_map(path + name + '_user_map.txt', user_map)
export_map(path + name + '_item_map.txt', item_map)

total_train = export_data(path + name + '_train.txt', train_users)
total_valid = export_data(path + name + '_valid.txt', valid_users)
total_test = export_data(path + name + '_test.txt', test_users)
print('total behaviors: ', total_train + total_valid + total_test)
