cd ../..
for model in it lstm itf FiLM
do
for seq_len in 5 10 20 40
do
for pred_len in 1 2 4
do
for e_layers in 2 3
do
for batch_size in 64 128
do
for learning_rate in 0.00005 0.00001 0.000005 0.000001 0.0000001
do
  python -u test_rbf.py \
  --e_layers $e_layers --buffer_size $batch_size --lr $learning_rate --model $model --seq_len $seq_len --pred_len $pred_len
done
done
done
done
done
done