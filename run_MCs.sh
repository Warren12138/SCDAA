#!/bin/bash

python Exercise1_2_functional.py Exercise1_2/value_MC/5x5/FSS_1e5/1_step 1 100000 &
python Exercise1_2_functional.py Exercise1_2/value_MC/5x5/FSS_1e5/10_steps 10 100000 &
python Exercise1_2_functional.py Exercise1_2/value_MC/5x5/FSS_1e5/50_steps 50 100000 &
python Exercise1_2_functional.py Exercise1_2/value_MC/5x5/FTSN_5e3/10_samples 5000 10 &
python Exercise1_2_functional.py Exercise1_2/value_MC/5x5/FTSN_5e3/50_samples 5000 50 &
python Exercise1_2_functional.py Exercise1_2/value_MC/5x5/FTSN_5e3/100_samples 5000 100 &
