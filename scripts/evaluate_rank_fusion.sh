export CUDA_VISIBLE_DEVICES=0
REPO_DIR=/query_variation_generators/disentangled_information_needs
source ${REPO_DIR}/env/bin/activate

OUT_DIR=${REPO_DIR}/data/
MAX_ITER=50

VARIATIONS_FILE_ANTIQUE=variations_antique_labeled.csv
VARIATIONS_FILE_TREC_DL=variations_trec2019_labeled.csv
VARIATIONS_FILE_DL_TYPO=variations_dl-typo_labeled.csv

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:antique/train/split200-valid' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_ANTIQUE

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:antique/train/split200-valid' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_ANTIQUE \
        --retrieval_model_name "BM25+RM3"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:antique/train/split200-valid' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_ANTIQUE \
        --retrieval_model_name "BM25+BERT" \
        --train_dataset "irds:antique/train" \
        --max_iter $MAX_ITER

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:antique/train/split200-valid' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_ANTIQUE \
        --retrieval_model_name "BM25+KNRM" \
        --train_dataset "irds:antique/train" \
        --max_iter $MAX_ITER

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:antique/train/split200-valid' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_ANTIQUE \
        --retrieval_model_name "BM25+T5"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:antique/train/split200-valid' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_ANTIQUE \
        --retrieval_model_name "https://macavaney.us/pt_models/msmarco.epic.seed42.tar.gz"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:antique/train/split200-valid' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_ANTIQUE \
        --retrieval_model_name "https://macavaney.us/pt_models/msmarco.convknrm.seed42.tar.gz"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
        --retrieval_model_name "BM25+RM3"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
        --retrieval_model_name "BM25+BERT" \
        --train_dataset "irds:msmarco-passage/train" \
        --max_iter $MAX_ITER

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
        --retrieval_model_name "BM25+KNRM" \
        --train_dataset "irds:msmarco-passage/train" \
        --max_iter $MAX_ITER

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
        --retrieval_model_name "BM25+T5"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
        --retrieval_model_name "https://macavaney.us/pt_models/msmarco.epic.seed42.tar.gz"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'irds:msmarco-passage/trec-dl-2019/judged' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_TREC_DL \
        --retrieval_model_name "https://macavaney.us/pt_models/msmarco.convknrm.seed42.tar.gz"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'dl-typo' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_DL_TYPO \

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'dl-typo' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_DL_TYPO \
        --retrieval_model_name "BM25+RM3"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'dl-typo' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_DL_TYPO \
        --retrieval_model_name "BM25+BERT" \
        --train_dataset "irds:msmarco-passage/train" \
        --max_iter $MAX_ITER

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'dl-typo' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_DL_TYPO \
        --retrieval_model_name "BM25+KNRM" \
        --train_dataset "irds:msmarco-passage/train" \
        --max_iter $MAX_ITER

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'dl-typo' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_DL_TYPO \
        --retrieval_model_name "BM25+T5"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'dl-typo' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_DL_TYPO \
        --retrieval_model_name "https://macavaney.us/pt_models/msmarco.epic.seed42.tar.gz"

python ${REPO_DIR}/disentangled_information_needs/evaluation/rank_fusion.py --task 'dl-typo' \
        --output_dir $OUT_DIR/ \
        --variations_file $OUT_DIR/$VARIATIONS_FILE_DL_TYPO \
        --retrieval_model_name "https://macavaney.us/pt_models/msmarco.convknrm.seed42.tar.gz"
