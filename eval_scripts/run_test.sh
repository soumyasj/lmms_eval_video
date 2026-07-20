# --- diagnostics: confirm which interpreter / torchcodec the run actually uses ---
echo "[diag] python  : $(which python)"
python -c "import sys, torch; print('[diag] exe    :', sys.executable); print('[diag] torch  :', torch.__version__)"
python -c "import torchcodec; print('[diag] tcodec :', torchcodec.__version__, torchcodec.__file__)"
python -c "from torchcodec.decoders import VideoDecoder; print('[diag] VideoDecoder import OK')"
# ---------------------------------------------------------------------------------

# without vllm
# python -m lmms_eval \
#   --model qwen2_5_vl \
#   --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=602112 \
#   --tasks videomme \
#   --batch_size 1 

# python -m lmms_eval \
#   --model vllm \
#   --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=602112,max_model_len=32768,gpu_memory_utilization=0.9,max_num_seqs=8,max_frame_num=32,limit_mm_per_prompt={\"video\":1} \
#   --tasks videomme \
#   --batch_size 8 

# python -m lmms_eval \
#   --model vllm \
#   --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=602112,max_model_len=32768,gpu_memory_utilization=0.9,max_num_seqs=8,max_frame_num=256,limit_mm_per_prompt={\"video\":1} \
#   --tasks video_mmmu \
#   --batch_size 8 

python -m lmms_eval \
  --model vllm \
  --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=602112,max_model_len=32768,gpu_memory_utilization=0.9,max_num_seqs=4,nframes=64,max_frame_num=16,limit_mm_per_prompt={\"video\":1} \
  --tasks video_mmmu \
  --batch_size 4


# with vllm
# python -m lmms_eval \
#   --model vllm \
#   --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=602112,max_model_len=32768,gpu_memory_utilization=0.9,max_num_seqs=8,max_frame_num=16,limit_mm_per_prompt={\"video\":1} \
#   --tasks mvbench_object_shuffle \
#   --batch_size 8
