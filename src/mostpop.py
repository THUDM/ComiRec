import math
from collections import defaultdict

class MostPopular:
    def __init__(self, source):
        self.train_graph = self.read(source + '_train.txt')
        self.test_graph = self.read(source + '_test.txt')

    def read(self, source):
        graph = {}
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                if user_id not in graph:
                    graph[user_id] = []
                graph[user_id].append(item_id)
        return graph
    
    def evaluate(self, N=50):
        item_count = defaultdict(int)
        for user in self.train_graph.keys():
            for item in self.train_graph[user]:
                item_count[item] += 1
        item_list = list(item_count.items())
        item_list.sort(key=lambda x:x[1], reverse=True)
        item_pop = set()
        for i in range(N):
            item_pop.add(item_list[i][0])
        total_recall = 0.0
        total_ndcg = 0.0
        total_hitrate = 0
        for user in self.test_graph.keys():
            recall = 0
            dcg = 0.0
            item_list = self.test_graph[user]
            item_list = item_list[int(len(item_list) * 0.8):]
            for no, item_id in enumerate(item_list):
                if item_id in item_pop:
                    recall += 1
                    dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
            total_recall += recall * 1.0 / len(item_list)
            if recall > 0:
                total_ndcg += dcg / idcg
                total_hitrate += 1
        total = len(self.test_graph)
        recall = total_recall / total
        ndcg = total_ndcg / total
        hitrate = total_hitrate * 1.0 / total
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}


if __name__ == "__main__":
    data = MostPopular('./data/book_data/book')
    # data = MostPopular('./data/taobao_data/taobao')
    print(data.evaluate(50))
