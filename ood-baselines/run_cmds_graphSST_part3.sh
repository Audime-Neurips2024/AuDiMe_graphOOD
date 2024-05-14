for dataset in "Graph-SST2" "Graph-Twitter";
do
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --eval_metric 'acc' --seed '[1,2,3]' --num_layers 4 --emb_dim 32 --pretrain 50 --batch_size 64 --dataset ${dataset} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 1e-3 --model 'gin' --dropout 0. --result_name 'grea0' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep' --eval_metric 'acc' --seed '[1,2,3]' --num_layers 4 --emb_dim 32 --pretrain 50 --batch_size 64 --dataset ${dataset} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 1e-5 --model 'gin' --dropout 0.  --result_name 'grea1' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --eval_metric 'acc' --seed '[1,2,3]' --num_layers 4 --emb_dim 32 --pretrain 50 --batch_size 64 --dataset ${dataset} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 1e-1 --model 'gin' --dropout 0.  --result_name 'grea2' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --eval_metric 'acc' --seed '[1,2,3]' --num_layers 4 --emb_dim 32 --pretrain 50 --batch_size 64 --dataset ${dataset} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 1 --model 'gin' --dropout 0.  --result_name 'grea3' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --eval_metric 'acc' --seed '[1,2,3]' --num_layers 4 --emb_dim 32 --pretrain 50 --batch_size 64 --dataset ${dataset} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 10 --model 'gin' --dropout 0.  --result_name 'grea4' &
wait
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --eval_metric 'acc' --seed '[1,2,3]' --num_layers 4 --emb_dim 32 --pretrain 50 --batch_size 64 --dataset ${dataset}  --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-5  --num_envs 3 --pretrain 50 -pe 10 --result_name 'gil0' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --eval_metric 'acc' --seed '[1,2,3]' --num_layers 4 --emb_dim 32 --pretrain 50 --batch_size 64 --dataset ${dataset}  --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-3  --num_envs 3 --pretrain 50 -pe 10 --result_name 'gil1' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --eval_metric 'acc' --seed '[1,2,3]' --num_layers 4 --emb_dim 32 --pretrain 50 --batch_size 64 --dataset ${dataset} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-1  --num_envs 3 --pretrain 50 -pe 10 --result_name 'gil2' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --eval_metric 'acc' --seed '[1,2,3]' --num_layers 4 --emb_dim 32 --pretrain 50 --batch_size 64 --dataset ${dataset}  --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1  --num_envs 3 --pretrain 50 -pe 10 --result_name 'gil3' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --eval_metric 'acc' --seed '[1,2,3]' --num_layers 4 --emb_dim 32 --pretrain 50 --batch_size 64 --dataset ${dataset}  --r -1 --contrast 0 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'gsat' --model 'gin' --dropout 0.  --result_name 'gsat' &
wait
done;

