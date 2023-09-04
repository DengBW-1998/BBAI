# UBAI

This repository contains the demo code of the paper: 

> Graph Adversarial Black-Box Attacks Based on Implicit Relations. 

If you have any question or you find any bug about codes, please contact me at wy727100600@163.com


Some code blocks are copied from the following papers:
- Bine: Bipartite network embedding. Gao M, Chen L, He X, et al. 
- Adversarial attacks on node embeddings via graph poisoning. Bojchevski A, Günnemann S.
- Adversarial Attacks and Defenses on Graphs. Jin W, Li Y, Xu H, et al.
- Adversarial attacks on graph neural networks via meta learning. Zügner D, Günnemann S.
- Adversarial attack on network embeddings via supervised network poisoning. Gupta V, Chakraborty T.
- Revisiting graph adversarial attack and defense from a data distribution perspective. Li K, Liu Y, Ao X, et al.
- Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs. Chaoyang He, Tian Xie, Yu Rong, et al.


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
- python test_pgdattack.py
- python test_viking.py
- python test_HA.py

If you want to calculate the parameter alpha of degree distribution:
- python alpha_calculation.py


## Important Hyperparameters

**Ablation Experience**
- n_flips*iteration is the actual num of perturbation

**Baseline Experience**
- the actual num of perturbation is set automatically