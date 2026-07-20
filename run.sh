export LMMS_EVAL_DATASETS_CACHE=/tmp/lmms_eval_hf_datasets_$USER

for i in $(seq 1 1);
do
    python -m lmms_eval \
        --model vllm \
        --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=602112,max_model_len=32768,gpu_memory_utilization=0.9,max_num_seqs=1,nframes=2,max_frame_num=2,limit_mm_per_prompt={\"video\":1} \
        --tasks videomme \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix qwen25vl_2frames_run${i}_mseq1 \
        --output_path ./logs/videomme_subset
done
