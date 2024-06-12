import os
from transformers import AutoModel
from numpy.linalg import norm
import numpy as np

class QueryEmoji:
    def __init__(self, feature_fn=None):
        if feature_fn is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            feature_fn = os.path.join(current_dir, 'emoji.npy')
        self.model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)
        emoji = np.load(feature_fn, allow_pickle=True).item()
        self.emoji_features = emoji['emoji_features']
        self.emoji_list = emoji['emoji_list']

    def query(self, description, n_candidates=5, return_score=False):
        query = self.model.encode(description)
        query = query / norm(query)
        scores = self.emoji_features @ query
        inds = np.argpartition(-scores, min(n_candidates, len(self.emoji_list)-1))[:n_candidates]
        if return_score:
            return self.emoji_list[inds].tolist(), scores[inds].tolist()
        else:
            return self.emoji_list[inds].tolist()

if __name__ == '__main__':
    import time
    E = QueryEmoji()
    def query_emoji_test(description, n_candidates=5):
        start = time.time()
        r = E.query(description, n_candidates)
        print(f'{description}:', " | ".join(r))
        # print("\ttime:", time.time() - start)
    while True:
        input_ = input('请输入描述：')
        if input_ == 'q' or input_ == '':
            break
        elif input_ == 'e':
            exit()
        query_emoji_test(input_, 10)
    query_emoji_test('开心')
    query_emoji_test('兴高采烈')
    query_emoji_test('伤心欲绝')
    query_emoji_test('鼠标')
    query_emoji_test('起重机')
    query_emoji_test('鹤')
    query_emoji_test('喝酒', 10)
    query_emoji_test('剪刀')
    query_emoji_test('黑人拳头')
    query_emoji_test('black person')
    query_emoji_test('a beautiful girl with a hat')
    query_emoji_test('a black cat')
    query_emoji_test('乌云密布的天空')
    query_emoji_test('月落乌啼霜满天')
    query_emoji_test('深蓝的天空中挂着一轮金黄的圆月')
    query_emoji_test('深蓝的天空中挂着一轮金黄的圆月，下面是海边的沙地，都种着一望无际的碧绿的西瓜。')
    query_emoji_test('其间有一个十一二岁的少年，项带银圈，手捏一柄钢叉，向一匹猹尽力地刺去')
    query_emoji_test('圆月')
    query_emoji_test('满月')
    query_emoji_test('汗牛充栋')
    query_emoji_test('吊民伐罪')
    query_emoji_test('sunny')
    query_emoji_test('sunny day')
    query_emoji_test('毕加索', 10)
    query_emoji_test('爱因斯坦', 10)
    query_emoji_test('画家', 10)
    query_emoji_test('科学家', 10)
    query_emoji_test('跳高')
    query_emoji_test('卫生纸')
    query_emoji_test('大学')
    query_emoji_test('论文')
    query_emoji_test('篮球')
    query_emoji_test('打篮球')
    query_emoji_test('打架')
    query_emoji_test('满分')
    query_emoji_test('100分')
    query_emoji_test('满分，100分')
    query_emoji_test('100元')
    query_emoji_test('无')
    query_emoji_test('nothing')
    query_emoji_test('reverse')
    query_emoji_test('intersection')
    query_emoji_test('8')
    query_emoji_test('8.6')
    query_emoji_test('3.14')
    query_emoji_test('3.14159')
    query_emoji_test('4点30')
    query_emoji_test('下午')
    query_emoji_test('晚上')
    query_emoji_test('晚上好')
    query_emoji_test('🍡')
    query_emoji_test('❤️')
    query_emoji_test('❤️🔵')
    query_emoji_test('a ❤️ in 🔵')
    query_emoji_test('📖')
    query_emoji_test('🔵📖')


