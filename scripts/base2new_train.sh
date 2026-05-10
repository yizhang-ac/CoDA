ROOT=/data
TRAINER=CoDA
FOLDER_PREFIX="output"

SHOTS=16
GPU=2

DATASETS=(imagenet)

for DATASET in "${DATASETS[@]}"; do
    ROOT="/data"
    CFG=${dataset_cfg_map[$DATASET]:-vit_b32_ep15_ctxv1}
    LOADEP=${load_ep_map[$DATASET]:-15}

    for SEED in 1 2 3; do
        # (1) base 阶段
        BASE_DIR=${FOLDER}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        if [[ -d "$BASE_DIR" ]]; then
            echo "[SKIP] base already exists: $BASE_DIR"
        else
            echo "[RUN ] base stage -> $BASE_DIR"
            CUDA_VISIBLE_DEVICES=$GPU python train.py \
                --root "$ROOT" \
                --seed "$SEED" \
                --trainer "$TRAINER" \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir "$BASE_DIR" \
                DATASET.NUM_SHOTS "$SHOTS" \
                DATASET.SUBSAMPLE_CLASSES base
        fi

        # (2) new 阶段
        NEW_DIR=${FOLDER}/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        if [[ -d "$NEW_DIR" ]]; then
            echo "[SKIP] new already exists: $NEW_DIR"
        else
            echo "[RUN ] new stage -> $NEW_DIR"
            CUDA_VISIBLE_DEVICES=$GPU python train.py \
                --root "$ROOT" \
                --seed "$SEED" \
                --trainer "$TRAINER" \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir "$NEW_DIR" \
                --model-dir "$BASE_DIR" \
                --load-epoch "$LOADEP" \
                --eval-only \
                DATASET.NUM_SHOTS "$SHOTS" \
                DATASET.SUBSAMPLE_CLASSES new
        fi
    done
done