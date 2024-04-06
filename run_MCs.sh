#!/bin/bash

python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FSS_1e5/1_step 1 100000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FSS_1e5/10_steps 10 100000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FSS_1e5/50_steps 50 100000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FSS_1e5/100_steps 100 100000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FSS_1e5/500_steps 500 100000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FSS_1e5/1000_steps 1000 100000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FSS_1e5/5000_steps 5000 100000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FTSN_5e3/10_samples 5000 10 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FTSN_5e3/50_samples 5000 50 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FTSN_5e3/100_samples 5000 100 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FTSN_5e3/500_samples 5000 500 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FTSN_5e3/1000_samples 5000 1000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FTSN_5e3/5000_samples 5000 5000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FTSN_5e3/10000_samples 5000 10000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FTSN_5e3/50000_samples 5000 50000 &
python lib/Exercise1_2_parallel_MC.py Exercise1_2/value_MC/1x1/FTSN_5e3/100000_samples 5000 100000 &
