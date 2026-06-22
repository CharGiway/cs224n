可以，自己手写一个最小版反而更容易理解。我们不用 `gensim`，直接实现一个简化版 **Skip-gram + Negative Sampling**。

为了看清楚，我用非常小的数据：

```text
我 喜欢 吃 苹果
我 喜欢 喝 茶
```

目标：

让模型学到：

```
苹果 ≈ 茶
```

（因为它们都出现在类似位置）

---

## 1. 准备数据

```python
import torch
import torch.nn as nn
import torch.optim as optim


text = [
    "我 喜欢 吃 苹果".split(),
    "我 喜欢 喝 茶".split()
]


# 建词表
words = []

for sentence in text:
    for w in sentence:
        if w not in words:
            words.append(w)


word2id = {
    w:i for i,w in enumerate(words)
}

id2word = {
    i:w for w,i in word2id.items()
}


vocab_size = len(words)

print(word2id)
```

输出：

```
{
 我:0,
 喜欢:1,
 吃:2,
 苹果:3,
 喝:4,
 茶:5
}
```

---

## 2. 构造 Skip-gram 样本

比如：

```
我 喜欢 吃 苹果
```

窗口=1

生成：

```
(我,喜欢)
(喜欢,我)
(喜欢,吃)
(吃,喜欢)
(吃,苹果)
(苹果,吃)
```

代码：

```python
pairs = []


window = 1


for sentence in text:

    ids = [
        word2id[w]
        for w in sentence
    ]


    for i,center in enumerate(ids):

        for j in range(
            max(0,i-window),
            min(len(ids),i+window+1)
        ):

            if i != j:
                pairs.append(
                    (center, ids[j])
                )


print(pairs)
```

输出：

```
[
(0,1),
(1,0),
(1,2),
(2,1),
(2,3),
(3,2)
...
]
```

这就是训练数据。

---

## 3. 定义模型

Skip-gram 本质：

输入：

```
苹果
```

输出：

```
吃 的概率
```

所以：

```python
class SkipGram(nn.Module):

    def __init__(self, vocab_size, dim):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            dim
        )


        self.linear = nn.Linear(
            dim,
            vocab_size
        )


    def forward(self,x):

        v = self.embedding(x)

        out = self.linear(v)

        return out
```

这里：

```python
embedding
```

就是我们最终想要的 word vector。

---

## 4. 训练

```python
model = SkipGram(
    vocab_size,
    dim=10
)


loss_fn = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)



for epoch in range(1000):

    total_loss = 0


    for center,target in pairs:


        center = torch.tensor(
            [center]
        )


        target = torch.tensor(
            [target]
        )


        pred = model(center)


        loss = loss_fn(
            pred,
            target
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


    if epoch % 100 == 0:
        print(
            epoch,
            total_loss
        )
```

---

## 5. 看训练出来的词向量

训练结束：

```python
vectors = model.embedding.weight


for i in range(vocab_size):

    print(
        id2word[i],
        vectors[i].detach()
    )
```

类似：

```
我
tensor([0.23,0.12,...])


喜欢
tensor([0.51,0.44,...])


苹果
tensor([-0.2,0.8,...])


茶
tensor([-0.18,0.75,...])
```

你会发现：

```
苹果
茶
```

向量距离会比较近。

---

这里最关键的一行：

```python
self.embedding = nn.Embedding(vocab_size, dim)
```

刚开始：

```
苹果:

[随机数字]
```

训练过程中：

每次预测错：

```
苹果 -> 吃
```

反向传播：

调整：

```
苹果的向量
吃的向量
```

慢慢让：

```
苹果
```

这个点移动到一个更合理的位置。

所以 Word2Vec 本质非常朴素：

**随机初始化一堆点，然后通过“预测上下文”不断移动这些点，最后空间结构变成语义结构。**

你之前提到“中心词预测周围词不符合直觉”，其实这个手写版看起来会更明显：它不是为了预测任务本身，而是利用预测误差作为“移动词向量”的信号。
