| Model | N | Accuracy | Balanced Accuracy | Macro-F1 | Depressed F1 | AUROC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|
| IDIAP text teacher - epoch 0 original | 34 | 0.8529 | 0.8676 | 0.8419 | 0.8000 | 0.7826 | 19 | 4 | 1 | 10 |
| USSD audio teacher - epoch 0 original | 34 | 0.8235 | 0.7747 | 0.7875 | 0.7000 | 0.6364 | 21 | 2 | 4 | 7 |
| v3 text branch | 34 | 0.7941 | 0.8241 | 0.7850 | 0.7407 | 0.8063 | 17 | 6 | 1 | 10 |
| v3 audio branch | 34 | 0.7059 | 0.6166 | 0.6222 | 0.4444 | 0.5257 | 20 | 3 | 7 | 4 |
| No-KD multimodal | 34 | 0.7353 | 0.7095 | 0.7043 | 0.6087 | 0.7470 | 18 | 5 | 4 | 7 |
| Standard KD | 34 | 0.7353 | 0.7569 | 0.7236 | 0.6667 | 0.7668 | 16 | 7 | 2 | 9 |
