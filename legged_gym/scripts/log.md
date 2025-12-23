## NOTE
```bash
conda activate go1_gym
```

## base
- 1223
```txt
原版 go2 奖励
 --resume  --load_run Oct29_10-17-44_ --checkpoint 7500
```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go1_trot --max_iterations 5000 --seed 1 --num_envs 4096 --run_name base --resume  --load_run Oct29_10-17-44_ --checkpoint 7500
```