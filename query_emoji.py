from transformers import AutoModel
from numpy.linalg import norm
import numpy as np

model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True) # trust_remote_code is needed to use the encode method

emoji = np.load('emoji.npy', allow_pickle=True).item()
emoji_features = emoji['emoji_features']
emoji_list = emoji['emoji_list']
emoji_code_list = emoji['emoji_code_list']

def query_emoji(description, n_candidates=5, verbose=False):
    query = model.encode(description)
    query = query / norm(query)
    scores = emoji_features @ query
    inds = np.argpartition(-scores, min(n_candidates, len(emoji_list)-1))[:n_candidates]
    if verbose:
        return emoji_list[inds].tolist(), emoji_code_list[inds].tolist(), scores[inds].tolist()
    else:
        return emoji_list[inds].tolist()

if __name__ == '__main__':
    import time
    def query_emoji_test(description, n_candidates=5):
        start = time.time()
        r = query_emoji(description, n_candidates)
        print(f'{description}:', " ".join(r))
        # print("\ttime:", time.time() - start)
    query_emoji_test('开心')
    query_emoji_test('兴高采烈')
    query_emoji_test('伤心欲绝')
    query_emoji_test('喝酒', 10)
    query_emoji_test('剪刀')
    query_emoji_test('黑人')
    query_emoji_test('black person')
    query_emoji_test('a beautiful girl with a hat')
    query_emoji_test('a black cat')
    query_emoji_test('乌云密布的天空')
    query_emoji_test('月落乌啼霜满天')
    query_emoji_test('深蓝的天空中挂着一轮金黄的圆月')
    query_emoji_test('深蓝的天空中挂着一轮金黄的圆月，下面是海边的沙地，都种着一望无际的碧绿的西瓜。')
    query_emoji_test('其间有一个十一二岁的少年，项带银圈，手捏一柄钢叉，向一匹猹尽力地刺去')
    query_emoji_test('sunny')
    query_emoji_test('sunny day')
    query_emoji_test('毕加索', 10)
    query_emoji_test('跳高')
    query_emoji_test('卫生纸')
    query_emoji_test('打篮球')
    query_emoji_test('打架')
    query_emoji_test('满分')
    query_emoji_test('100分')
    query_emoji_test('满分，100分')
    query_emoji_test('100元')
    query_emoji_test('无')
    query_emoji_test('nothing')
    query_emoji_test('8')
    query_emoji_test('8.8')
    query_emoji_test('4点30')

