# BBAI

This repository contains the demo codes of the paper: 

> Bipartite Graph Black-Box Adversarial Attacks Based on Implicit Relations. 

If you have any question or you find any bug about codes, please contact me at wy727100600@163.com


## Environment settings

- python==3.6.13
- numpy==1.19.2
- deeprobust==0.2.4
- pytorch==1.10.1
- tensorflow==2.6.0
- gensim==3.8.3


## Basic Usage

**Usage**

If you want to run BBAI：
- python BBAI.py

If you want to play ablation experiments:
- python BBAI_exp_abs.py
- python BBAI_exp_app.py
- python BBAI_imp_abs.py

If you want to play baselines:
- python cln.py
- python rndAttack.py
- python test_dice.py
- python test_metattack.py
- ...

If you want to calculate the parameter alpha of degree distribution:
- python alpha_calculation.py

If you want to check the experiments in INTRPDUCTION:
- python imp_relation_intro.py


## Important Hyperparameters

- If you want to run the ablation experiment, set linkpre_abl to True,else to False.
- If you want to read flipped matrix directly, set read_dir to True. You can't set it to True in the first running except for Pubmed under Metattack.
