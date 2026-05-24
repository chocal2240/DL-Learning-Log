# Tokenizer Scripts

这里保留 SentencePiece 分词器训练脚本。

生成的 `eng.model`、`eng.vocab`、`chn.model` 和 `chn.vocab` 属于本地训练产物，不放入学习日志仓库。准备好 `data/corpus.en` 和 `data/corpus.ch` 后可重新生成：

```bash
python tokenizer/tokenize.py
```
