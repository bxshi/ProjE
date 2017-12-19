## ProjE

ProjE: Embedding Projection for Knowledge Graph Completion

Accepted at AAAI'2017.

**Please also check our latest work (Open-World Knowledge Graph Completion) accepted at AAAI2018: https://github.com/bxshi/ConMask**

---

### How to run the code

First, please download the data file from https://github.com/thunlp/KB2E and unzip the `data.zip` file. You will get two folders `FB15k` and `WN18`. 


You can call `ProjE` using

```bash
./ProjE_softmax.py --dim 200 --batch 200 --data ./data/FB15k/ --eval_per 1 --worker 3 --eval_batch 500 --max_iter 100 --generator 10
```


