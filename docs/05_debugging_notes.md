为什么SFT模型对格式遵循效果这么差？

你的 tokenizer：

```python
tokenizer.encode("<think>")
```

输出：

```
[151667]
```

```python
tokenizer.encode("</think>")
```

输出：

```
[151668]
```

说明：

✅ `<think>` 和 `</think>` **已经是单独的 special token**

这个不是问题。

但是：

```python
tokenizer.encode("<answer>")
```

输出：

```
[27, 9217, 29]
```

说明：

❌ `<answer>` **不是 special token**

它被拆成：

```
<
answer
>
```

三个普通 token。

---

## 这个解释了你“think格式有时正常、answer格式基本没有”的可能性

你的目标格式：

```
<think>
reasoning
</think>

<answer>
3
</answer>
```

其中：

| token       | 状态              |
| ----------- | --------------- |
| `<think>`   | special token ✅ |
| `</think>`  | special token ✅ |
| `<answer>`  | 普通token ❌       |
| `</answer>` | 未知（建议检查）        |

模型实际上学习的是：

```
特殊符号
<think>
    reasoning
特殊符号
</think>

普通字符串
<
answer
>
3
<
/
answer
>
```

对于 0.6B 小模型：

这会造成明显困难。

---

# 为什么 `<think>` 能学会，而 `<answer>` 学不会？

因为 `<think>` 在 Qwen3 tokenizer 里本身可能已经存在。

Qwen3 原生支持 reasoning token：

例如：

```
<think>
...
</think>
```

所以：

```
151667
151668
```

对应的是模型预训练期间已经见过的语义单元。

模型知道：

```
151667
=
开始思考区域
```

而 `<answer>`：

```
27 9217 29
```

只是三个字符组合。

预训练阶段：

模型可能见过：

```
<answer>
```

但它不是一个整体概念。

---

# 这也解释了你的实验结果

你的结果：

数学能力：

```
GSM8K pass@1 42.8%
MATH pass@1 31.6%
```

很好。

说明：

模型学会：

```
问题
↓
<think>
推理
</think>
答案
```

但是：

严格：

```
<think>
...
</think>

<answer>
...
</answer>
```

失败。

因为模型倾向：

生成：

```
<think>
...
</think>

The answer is 3
```

或者：

```
<think>
...
</think>

3
```

而不是：

```
<answer>
3
</answer>
```

---

# 你现在应该做的修改

## 方案1（推荐）：把 answer 也加入 special token

这是最干净的。

修改 tokenizer：

```python
special_tokens_dict = {
    "additional_special_tokens": [
        "<answer>",
        "</answer>"
    ]
}

num_added_tokens = tokenizer.add_special_tokens(
    special_tokens_dict
)

model.resize_token_embeddings(len(tokenizer))
```

然后重新SFT。

重新检查：

```python
tokenizer.encode("<answer>")
```

应该变成：

```
[151669]
```

类似：

```
<think>  [151667]
</think> [151668]

<answer> [151669]
</answer>[151670]
```

---

# 但是这里有一个细节

你现在用的是：

```
Qwen3-0.6B-Base
```

如果你自己添加：

```
<answer>
```

这个token：

embedding是随机初始化的。

所以：

刚开始：

```
<answer>
```

没有任何意义。

但是你的SFT数据里面有：

3800个样本：

每个样本都有：

```
<answer>
数字
</answer>
```

所以很快会学会。

对于600M模型：

3800样本完全够。

---

# 方案2：不用answer标签，使用Qwen3原生格式

Qwen3其实更推荐：

```
<think>
...
</think>

最终答案
```

因为：

```
<think>
```

是模型原生reasoning机制。

很多reasoning模型并不用：

```
<answer>
```

而是：

```
<think>
reasoning
</think>
final
```

例如：

```
<think>
2+3=5
</think>
5
```

这种对于Qwen3更自然。

---

# 我个人更建议你的实验改成：

SFT目标：

```
<think>
...
</think>
The answer is:
3
```

或者：

```
<think>
...
</think>
3
```

然后RL阶段：

reward解析最后答案。

原因：

你的目标是训练reasoning能力，不是训练XML格式。

---

# 另外检查一个东西

请你再跑：

```python
print(tokenizer.encode("</answer>"))
```

我预计：

大概率也是：

```
[27, xxx, 29]
```

如果两个answer tag都不是special token：

那么你的格式遵循失败几乎可以解释。

---

# 还有一个值得注意的点

你的system prompt：

> For every problem, first think step by step inside `<think>...</think>`

其实对于Qwen3-base来说：

可能是多余甚至干扰。

因为Qwen3已经有：

```
<think>
```

机制。

更好的cold start数据：

可以简化成：

system:

```
You are a math reasoning assistant.
```

assistant:

```
<think>
...
</think>
<answer>
...
</answer>
```

让模型从输出分布中学习，而不是靠instruction触发。

---

所以现在你的实验下一步非常明确：

1. ✅ 保留 `<think>`（已经是special token）
2. ✅ 添加 `<answer>` 和 `</answer>` special token
3. 重新SFT（不需要7 epoch，3~5 epoch即可）
4. 再测格式遵循率

我预计你的格式遵循率会有明显提升。

你这个问题其实已经定位到了 reasoning model 训练里一个非常核心的细节：**reasoning token design 和 output protocol tokenization**。这也是很多人复现 DeepSeek-R1 cold start 时容易踩的坑。
