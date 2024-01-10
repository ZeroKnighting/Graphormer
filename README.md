# ⚡️ Quickstart

### test
run `python graphormer/modules/multi_graph_utils.py` to test

### model training for MatBench
```
fairseq-train --user-dir ../graphormer  \
--MatBench-name matbench_dielectric \
--best-checkpoint-metric loss \
   --num-workers 0 --ddp-backend=c10d \
   --task MatBench --criterion MAE_loss --arch graphormer3d_base  \
   --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 5 \
   --lr-scheduler polynomial_decay --lr 3e-4 --warmup-updates 10000 \
   --total-num-update 1000000 --batch-size 4 \
   --dropout 0.0 --attention-dropout 0.1 \
    --tensorboard-logdir ./tsbs \
   --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 \
   --max-update 1000000 --log-interval 100 --log-format simple \
   --save-interval-updates 5000 --validate-interval-updates 2500 --keep-interval-updates 30 --no-epoch-checkpoints  \
   --save-dir ./ckpt --layers 12 --blocks 4 --required-batch-size-multiple 1  \
   --node-loss-weight 15
```
