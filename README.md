# query_emoji
Query emoji with natural language, Chinese or English.

## Install
You should first `git clone --depth 1 https://github.com/guo-yong-zhi/query_emoji` and run `pip install -r requirements.txt`. Then you can run `python query_emoji.py` to see the result. You can also use it in your code:
```python
from query_emoji import QueryEmoji
E = QueryEmoji()
emoji_candidates = E.query("a beautiful girl with a hat", 5)
print(emoji_candidates)
```
```
['🤠', '👧', '👸', '👒', '👧🏽']
```

## Examples
- 开心: 🙋 | 😀 | 😌 | 😆 | 😄
- 兴高采烈: 🥂 | 🍻 | 🎆 | 🎊 | 😁
- 伤心欲绝: 😦 | 😧 | 😭 | 😞 | 😢
- 鼠标: 🕹 | 🖱 | 🐭 | 🎮 | 🐁
- 起重机: 🏋 | 🏗 | 🚚 | 🕴 | 🚟
- 鹤: 🐢 | 🐅 | 🐄 | 🗻 | 🐂
- crane: 🚁 | 🐓 | 🚂 | 🚟 | 🐦
- 喝酒: 🍷 | 🍸 | 🍻 | 🍺 | 🥃 | ☕️ | 🍾 | 🍇 | ☕ | 🍶
- 剪刀: ✂ | 💇 | 🔪 | ✂️ | 🍴
- 黑人拳头: ✊🏿 | 👨🏿 | 💪🏿 | 💇🏿 | 🙇🏿
- black person: 👨🏿 | 🙎🏿 | 👩🏿 | 🚶🏿 | 👱🏿
- a beautiful girl with a hat: 👧 | 🤠 | 👸 | 👒 | 👧🏽
- a black cat: 👦🏿 | 🐈 | 😹 | 😼 | 👨🏿
- 乌云密布的天空: ☁ | ☁️ | ⛅ | 🌦 | 🌧
- 月落乌啼霜满天: ⛱ | ❄ | 🙍🏾 | 🌕 | 🙇🏾
- 深蓝的天空中挂着一轮金黄的圆月: 🔵 | 🌙 | 🌝 | 🌘 | 🌇
- 深蓝的天空中挂着一轮金黄的圆月，下面是海边的沙地，都种着一望无际的碧绿的西瓜。: 🍉 | 🍆 | 🌝 | 🍌 | 🍈
- 其间有一个十一二岁的少年，项带银圈，手捏一柄钢叉，向一匹猹尽力地刺去: 🤠 | 👱🏿 | 🔧 | 🔦 | 👴🏿
- 圆月: 🌙 | 🌝 | 🌚 | ☪ | 🌜
- 满月: 🌕 | 🌝 | 🌙 | 🌛 | 🌜
- 汗牛充栋: 💩 | 😲 | 🐮 | 😯 | 🐴
- 吊民伐罪: 💀 | 🚒 | 🛥 | 🗄 | ⚰
- sunny: 🌞 | 🌤 | ☀ | ⛅ | ⛅️
- sunny day: 🌞 | ⛅ | 🌤 | ☀ | ⛅️
- 毕加索: ♏ | ♍ | ♌ | 🗻 | ✡ | ♑ | 🕋 | 🗿 | 🔬 | 🔭
- 爱因斯坦: ♒ | 🔬 | ♑ | ♏ | ♌ | ✡ | ♍ | 🏎 | ☦ | 🏛
- 画家: 🎨 | 🖌 | 🖍 | 🖼 | 🤔 | 👁 | ⛷ | 🗿 | 🏌 | 🚵
- 科学家: 🔬 | 🔭 | ♍ | ♏ | ♒ | ♊ | 👽 | ♑ | ♌ | 🍯
- 跳高: 🎚 | ✋ | 🏋 | 🕴 | 👠
- 卫生纸: 🇳🇷 | 🆕 | 🗞 | 🚻 | 🚽
- 大学: 🏛 | 🛐 | 🏫 | 🚄 | 🏢
- 论文: 🔬 | ⚖ | 🔖 | 🤔 | 🗞
- 篮球: 🏀 | ⚽ | ⚾️ | ⛹ | ⚾
- 打篮球: ⛹ | ⚾️ | 🏀 | 🏐 | ⚾
- 打架: 🤹 | ⚔ | 🤦 | 🔫 | 👊
- 满分: 🈵 | 🎖 | 🈳 | 🌂 | 🆎
- 100分: 💯 | 💶 | 🆓 | 💮 | ◽
- 满分，100分: 🈵 | 💶 | 💯 | 🈳 | 🎖
- 100元: 💶 | 💯 | 🍶 | 💴 | 💷
- 无: 🈚 | ☎ | 🈳 | 0️⃣ | ⛔
- nothing: 🈚 | 0️⃣ | 🈳 | 🚱 | ⛄️
- reverse: ⏪ | 🙃 | ↙ | 👇🏿 | 🔀
- intersection: 🛣 | ⏹ | 🚥 | 🚇 | 🚸
- 8: 8️⃣ | 🕗 | ✴️ | ✴ | 9️⃣
- 8.6: 6️⃣ | 6⃣️ | 8️⃣ | 🕕 | 🍿
- 3.14: 🕞 | 🕒 | 🕟 | 🈷️ | 🏵
- 3.14159: 🏵 | 🏃🏾 | 🐏 | 🆕 | 🌫
- 4点30: 🕟 | 🕞 | 🕡 | 🕠 | 🕓
- 下午: 🛰 | 🌇 | ⛅ | 🌆 | 👓
- 晚上: 🌃 | 🌇 | 🕶 | 🌉 | 🔭
- 晚上好: 🌃 | 🌇 | 🕶 | 🌠 | 🌉

---

I would extend my thanks to [emoji2vec](https://github.com/uclnlp/emoji2vec) and [jinaai](https://huggingface.co/jinaai/jina-embeddings-v2-base-zh).
