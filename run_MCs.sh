#!/bin/bash
source ../env_SCDAA/bin/activate
#python Exercise1_2_parallel_MC.py Exercise1_2/value_MC/5x5/FSS_1e5/5000_steps 5000 100000 &
#python Exercise1_2_parallel_MC.py Exercise1_2/value_MC/5x5/FTSN_5e3/50000_samples 5000 50000 &
python Exercise1_2_parallel_MC.py Exercise1_2/value_MC/5x5/FTSN_5e3/100000_samples 5000 100000 &
