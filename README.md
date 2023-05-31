# UBAI

This repository contains the demo code of the paper: 

> Bipartite Graph Adversarial Attacks Based on Implicit Relations. 

If you have any question or you find any bug about codes, please contact me at xxx(anonymous now).


Some code blocks copy from the following papers:
- Bine: Bipartite network embedding. Gao M, Chen L, He X, et al. 
- Adversarial attacks on node embeddings via graph poisoning. Bojchevski A, Günnemann S.
- Adversarial Attacks and Defenses on Graphs. Jin W, Li Y, Xu H, et al.
- Adversarial attacks on graph neural networks via meta learning. Zügner D, Günnemann S.

## Environment settings

- python==3.6.13
- numpy==1.19.2
- deeprobust==0.2.4
- pytorch==1.10.1
- tensorflow==2.6.0
- gensim==3.8.3



## Basic Usage

**Usage**

If you want to run UBAI：
- python UBAI.py

If you want to play ablation experiments:
- python UBAI_exp_abs.py
- python UBAI_exp_app.py
- python UBAI_imp_abs.py

If you want to play baselines:
- python cln.py
- python rndAttack.py
- python test_dice.py
- python test_metattack.py
- python test_pgdattack.py


**Main Hyperparameters:**

You can adjust hyperparameters in the source code files directly.

- Changing datasets: Copy the file "UBAI/codes/dblp/testModel.py" or "UBAI/codes/wiki/testModel.py" into the dirctory "UBAI/codes"

- You can adjust the rate of subgraph in "testModel.py".

- You can adjust other hyperparameters in "UBAI.py"(also "UBAI_exp_abs.py" or "test_metattack.py" etc.)
