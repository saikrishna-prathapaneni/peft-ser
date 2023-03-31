
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# for model_type in whisper_tiny whisper_base whisper_small; do
# for model_type in whisper_large; do
# for model_type in wav2vec2_0; do
for model_type in whisper_base; do
    for dataset in iemocap msp-improv crema_d; do
        for finetune_method in embedding_prompt; do
            for embedding_prompt_dim in 5; do
                CUDA_VISIBLE_DEVICES=0, taskset -c 1-20 python3 finetune_emotion.py --pretrain_model $model_type --dataset $dataset --learning_rate 0.0005 --num_epochs 30 --finetune_method $finetune_method --embedding_prompt_dim $embedding_prompt_dim
            done
        done
    done
done