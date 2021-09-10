# EaR-SCL
## Introduction
Paper:

The code is modified based on ELECTRA: https://github.com/google-research/electra.

## Fine-tuning on GLUE (ELECTRA):
### 1.environment:
CUDA 10 + tf 1.5 + python 3.6.8
### 2.requirements:
numpy

scipy

sklearn

six>=1.12.0
### run:
python3 run_finetuning.py --data_dir ./data/CoLA --electra_model ./electra_base --epochs 3 --use_cl true --c_type token_cutoff --cut_rate 0.01 --alpha 0.1 --tau 0.1  --write true --output_dir ./output/CoLA/token/a0.1_c0.01_t0.1 --hparams json/cola.json

## GLUE results: 
GLUE leaderboard: https://gluebenchmark.com/leaderboard

submitted name: 

1: ELECTRA-Large-NewSCL(single)

2: ELECTRA-Base-NewSCL(single)

