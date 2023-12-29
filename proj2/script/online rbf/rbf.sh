cd ../..
for model in itf tsf
do
for seq_len in 5 10
do
for pred_len in 1 3
do
for buffer_size in 32 64 128 256
do
  for d_model in 8 16 32 64
do
    for e_layers in 1 2
do
for batch_size in 8 16 32 64
do
for learning_rate in  0.0001 0.00005
do
  python -u testtt.py \
  --buffer_size $buffer_size --batch_size $batch_size --lr $learning_rate --model $model --seq_len $seq_len --pred_len $pred_len --d_model $d_model --e_layers $e_layers
done
done
done
done
done
done
done
done