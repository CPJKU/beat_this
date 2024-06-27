#!/bin/bash

# Runs all experiments listed at the bottom. Each experiment consists of a
# given number of repetitions using a particular name and arguments.
# Each single repetition checks if it was already run or is currently being
# run, creates a lockfile and trains the network. To distribute runs between
# multiple GPUs, run this script multiple times with different
# CUDA_VISIBLE_DEVICES. To distribute runs between multiple hosts, run this
# script multiple times with a shared output directory (via NFS).

here="${0%/*}"
outdir="$here/../JBT"

train_seed() {
	name="$1"
	seed="$2"
	fold=
	for arg in "${@:3}"; do
		if [[ "$arg" == "--fold="* ]]; then
			fold="${arg#*=}"
			break
		fi
	done
	lockdir="$outdir/$name"
	mkdir -p "$lockdir"
	if [ -z "$fold" ]; then
		echo "$name, seed=$seed"
		lockfile="$lockdir/$seed.lock"
	else
		echo "$name, seed=$seed, fold=$fold"
		lockfile="$lockdir/$seed.$fold.lock"
	fi
	if [ ! -f "$lockfile" ]; then
		echo "$HOSTNAME: $CUDA_VISIBLE_DEVICES" > "$lockfile"
		JBT_SEED=$seed python3 "$here"/train.py --wandb_log --comment "$name" "${@:3}" && echo "success" >> "$lockfile" || echo "failed" >> "$lockfile"
	fi
}

train_seeds() {
	name="$1"
	seeds="$2"
	for (( seed=0; seed<$seeds; seed++ )); do
		train_seed "$name" "$seed" "${@:3}"
	done
}

train() {
    name="$1"
    train_seeds "$name" 3 "${@:2}"
}

train_cv() {
	name="$1"
	for (( fold=0; fold<8; fold++)); do
		train_seed "$name" 0 --fold=$fold "${@:2}"
	done
}

# common settings
input="--input_enc=wav_mel(128,30,11000,log1pfix,3,bn,1024) --max_train_len=1500 --sample_rate=22050 --hop_size=441 --sampler=none"
augments="--augmentations=pitch_or_tempo(-5,+6,20,4) --frontend_augmentations=mask(permute,1.0,1,6,0.1,2.0,5,10)"
model="--n_heads=16 --compile transformer_encoder,frontend_final --task_heads=beat,downbeat:HSumBeat(Linear) --extra_frontend=CustomConv2d(1,C,32,4,3,s,4,BN,gelu,F,1,0.1,T,1,0.1,C,64,2,3,s,2,BN,gelu,F,2,0.1,T,2,0.1,C,128,2,3,s,2,BN,gelu,F,4,0.1,T,4,0.1,C,256,2,3,s,2,BN,gelu,L)"
loss="--loss=beat:BCE_pos(3,0);beat:BCE_neg(3,6);downbeat:BCE_pos(3,0);downbeat:BCE_neg(3,6) --compute_pos_weight --widen_target_mask_loss=7"
training="--val_frequency=5 --batch_size=8 --accumulate_grad_batches=8 --lenght_based_oversampling_factor 0.75 --max_epochs=100 --lr 0.0008 --optimizer=warm-split-adamw --weight_decay=0.01"

# final model
train "final" $input $augments $model $loss $training

# final model with 8-fold CV
train_cv "final-cv" $input $augments $model $loss $training

# final model, no val
train "final-noval" $input $augments $model $loss $training --val_datasets=""

# final model, no val, post-swa
swa="--max_epochs=110 --lr=0.0004 --optimizer=split-swa-adamw --save_every=1 --save_average=100"
train_seed "final-noval-swa" 0 $input $augments $model $loss $training $swa --val_datasets="" --finetune_checkpoint JBT/m7mk6pta/checkpoints/epoch=99-step=20700.ckpt
train_seed "final-noval-swa" 1 $input $augments $model $loss $training $swa --val_datasets="" --finetune_checkpoint JBT/oh69dpet/checkpoints/epoch=99-step=20700.ckpt
train_seed "final-noval-swa" 2 $input $augments $model $loss $training $swa --val_datasets="" --finetune_checkpoint JBT/fsyt3udn/checkpoints/epoch=99-step=20700.ckpt

# removing pitch augmentation
train "no-pitch" $input $augments $model $loss $training "--augmentations=tempo(20,4)"

# removing tempo augmentation
train "no-tempo" $input $augments $model $loss $training "--augmentations=pitch(-5,+6)"

# removing span masking
train "no-spanmask" $input $augments $model $loss $training "--frontend_augmentations="

# using masked BCE instead of shift-tolerant BCE
train "masked-bce" $input $augments $model $training "--loss=beat:BCE_pos(0,0);beat:BCE_neg(0,3);downbeat:BCE_pos(0,0);downbeat:BCE_neg(0,3)" --widen_target_mask_loss=7 --compute_pos_weight

# using plain BCE
train "plain-bce" $input $augments $model $training "--loss=beat:BCE_pos(0,0);beat:BCE_neg(0,0);downbeat:BCE_pos(0,0);downbeat:BCE_neg(0,0)" --widen_target_mask_loss=0 --compute_pos_weight

# using plain BCE and no pos weights
train "plain-bce-no-posweight" $input $augments $model $training "--loss=beat:BCE_pos(0,0);beat:BCE_neg(0,0);downbeat:BCE_pos(0,0);downbeat:BCE_neg(0,0)" --widen_target_mask_loss=0 --manual_pos_weight=beat:1,downbeat:1

# using standard task heads
train "no-hsum" $input $augments $model $loss $training --task_heads="*:Linear"

# removing the frontend transformers
train "no-frontend-tf" $input $augments $model $loss $training --extra_frontend="CustomConv2d(1,C,32,4,3,s,4,BN,gelu,C,64,2,3,s,2,BN,gelu,C,128,2,3,s,2,BN,gelu,C,256,2,3,s,2,BN,gelu,L)"

# using the 80bins frontend
train "80bins" $input $augments $model $loss $training --input_enc="wav_mel(80,30,11000,log1pfix,3,bn,1024)" --extra_frontend="CustomConv2d(1,C,64,3,3,MP,3,gelu,D,0.1,C,128,12,1,MP,3,gelu,D,0.1,C,512,3,3,MP,3,gelu)"

# without oversampling
train "no-oversampling" $input $augments $model $loss $training --lenght_based_oversampling_factor=0 --max_epochs=286

# (only use the Hung et al datasets?)
train "only-hung" $input $augments $model $loss $training --train_datasets=hung
train "only-hung-noval" $input $augments $model $loss $training --train_datasets=hung --val_datasets=""


# small model experiments
train "small h128" $input $augments $model $loss $training --n_heads=4 --n_hidden=128 --dim_feedforward=512
train "small h128-noval" $input $augments $model $loss $training --n_heads=4 --n_hidden=128 --dim_feedforward=512 --val_datasets=""