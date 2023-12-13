cd ../..
for model in itf
do
for seq_len in 5 10 20 40
do
for pred_len in 1 3 8
do
for buffer_size in 200 500 1000 2000 4000
do
for batch_size in 16 32 64 128
do
for learning_rate in 0.05 0.01 0.001 0.0001 0.00001
do
  python -u testtt.py \
  --buffer_size $buffer_size --buffer_size $batch_size --lr $learning_rate --model $model --seq_len $seq_len --pred_len $pred_len
done
done
done
done
done
done