cd ../..
for model in itf
do
for seq_len in 10 
do
for pred_len in 3 
do
for buffer_size in 64 
do
for batch_size in 64 
do
for learning_rate in 0.0004
do
  python -u testtt.py \
  --buffer_size $buffer_size --batch_size $batch_size --lr $learning_rate --model $model --seq_len $seq_len --pred_len $pred_len
done
done
done
done
done
done
