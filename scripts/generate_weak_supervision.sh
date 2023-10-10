REPO_DIR=/content/drive/MyDrive/REIT4841/query_variation_generators

export CUDA_VISIBLE_DEVICES=0
source ${REPO_DIR}/env/bin/activate

OUT_DIR=${REPO_DIR}/data/

for TASK in 'msmarco-passage/trec-dl-2019/judged' 'antique/train/split200-valid' 'dl-typo'
do
    python ${REPO_DIR}/examples/generate_weak_supervision.py --task $TASK \
        --output_dir $OUT_DIR 
done
