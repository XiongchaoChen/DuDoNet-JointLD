python train.py \
--experiment_name 'train_1' \
--model_type 'model_cnn' \
--data_root './' \
--norm 'BN' \
--net_filter 32 \
--n_denselayer 6 \
--growth_rate 32 \
--lr_1 1e-3 \
--lr_2 1e-3 \
--lr_3 1e-3 \
--step_size 1 \
--gamma 0.99 \
--n_epochs 200 \
--batch_size 1 \
--eval_epochs 5 \
--snapshot_epochs 5 \
--gpu_ids 0