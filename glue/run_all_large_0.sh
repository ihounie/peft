model="roberta-large"
for rank in 16 8
do
    for task in "sst2" "mrpc" "qnli" 
    do
        CUDA_VISIBLE_DEVICES=0 python run_glue.py configs/${model}_${task}_r=${rank}.json
    done
done