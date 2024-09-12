MODEL_PATH="../model/SFT/final_checkpoint"
DATA_PATH="../data/ultrafeedback/rm"
OUTPUT_DIR="../model/DPO/input_tuning_prefix_MSE_retain"

python prefix_tuning.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --use_neg True \
    --use_retain True \
    --use_cosine False