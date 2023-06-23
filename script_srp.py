import os
import numpy

seed = [111, 222, 333, 444, 555, 666, 777, 888, 999]

for i in range(9):
  os.system(f"python3 train_search.py --warmup 20 --prune_epochs 30 --prune_op_num 4 --dropout 0.2 --infer_test_portion 0.5 --method SRP-DARTS --seed {seed[i]}")
  #os.system(f"python3 train_search.py --warmup 20 --stage_epochs 30 --prune_op_num 4 --dropout 0.2 --infer_test_portion 0.5 --seed {seed[i]}")



