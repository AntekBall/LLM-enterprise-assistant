Retrieval on 80 questions with a known source

| Retrieval | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| dense | 0.55 | 0.70 | 0.79 | 0.88 | 0.653 |
| dense+rerank | 0.81 | 0.90 | 0.91 | 0.95 | 0.857 |
| hybrid | 0.55 | 0.75 | 0.82 | 0.88 | 0.662 |
| hybrid+rerank | 0.81 | 0.90 | 0.93 | 0.95 | 0.859 |

Answer quality (80 answerable, 19 unanswerable)

| Run | Correct | Correct (source in context) | Correct (answered only) | False refusals | Correct refusals | Faithful | Cited |
|---|---|---|---|---|---|---|---|
| Baseline (prompt v1) | 0.75 | 0.79 | 0.88 | 0.15 | 0.95 | 0.97 | 1.00 |
| Prompt v2 | 0.84 | 0.88 | 0.89 | 0.06 | 0.95 | 0.91 | 0.99 |
| Fine-tuned v1 | 0.74 | 0.81 | 0.92 | 0.20 | 1.00 | 0.94 | 1.00 |
| Fine-tuned v2 | 0.86 | 0.90 | 0.93 | 0.07 | 0.95 | 0.93 | 0.99 |
