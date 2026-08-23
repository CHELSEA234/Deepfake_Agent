# Dataset construction

| Expert | Historical format | Image | Description | Score |
| --- | --- | --- | --- | --- |
| FakeShield / image editing | JSONL | `image` | `outputs` | `score` or `p_yes` |
| M2F2-Det / face swap | JSONL | `image` | `text` | `score` |
| LLaVA semantic output | CSV | `filepath` | `answer` | `p_yes` |

The output schema is:

```json
{
  "id": "deterministic sha256",
  "image": "dataset-relative/image.png",
  "conversations": [
    {"from": "human", "value": "<image>\n...three expert reports..."},
    {"from": "gpt", "value": "fake"}
  ]
}
```

IDs are hashes of the image reference and conversations. Input order does not
affect output order.

## Datasets

No benchmark images, metadata, or generated reports are distributed here.
Refer to [FakeShield](https://github.com/zhipeixu/FakeShield) for the benchmark
list and dataset setup, and obtain each dataset from its official source with
roots configured outside this repository.

Dataset terms vary and some restrict redistribution.
