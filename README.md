# EaR-SCL
Paper:

The code is modified based on ELECTRA: https://github.com/google-research/electra.

## fine-tuning on GLUE (ELECTRA):

CUDA_VISIBLE_DEVICES=0 python3 run_finetuning.py \
        --data_dir ./data/CoLA \
        --electra_model ./electra_base \
        --epochs 3 \
        --use_cl true \
        --c_type token_cutoff \
        --cut_rate 0.01 \
        --alpha 0.1 \
        --tau 0.1 \
        --write true \
        --output_dir ./output/CoLA/a0.1_c0.01_t0.1 \
        --hparams json/cola.json

## GLUE results: 
GLUE leaderboard: https://gluebenchmark.com/leaderboard

submitted name: ELECTRA-Large-NewSCL(single)
                ELECTRA-Base-NewSCL(single)

