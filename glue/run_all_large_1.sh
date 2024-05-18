model="roberta-large"
for rank in 16 8
do
    for task in "rte" "stsb" "cola"
    do
        CUDA_VISIBLE_DEVICES=1 python run_glue.py configs/${model}_${task}_r=${rank}.json
    done
done