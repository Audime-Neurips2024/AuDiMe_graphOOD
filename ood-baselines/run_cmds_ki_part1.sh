
for dataset in "drugood_lbap_core_ki_size" "drugood_lbap_core_ki_assay" "drugood_lbap_core_ki_scaffold";
do
CUDA_VISIBLE_DEVICES=2 python main.py --eval_metric 'auc' --r -1 --num_layers 4  --batch_size 64 --emb_dim 32 --model 'gin' --pooling 'sum' -c_dim 32 --dataset ${dataset} --seed '[1,2,3]' --dropout 0.0   --contrast 0 -c_in 'raw'  -c_rep 'rep'  --spu_coe 0 --erm --irm_opt 'irm' --num_envs 2 --irm_p 1e-2 --result_name 'irm0' &
CUDA_VISIBLE_DEVICES=2 python main.py --eval_metric 'auc' --r -1 --num_layers 4  --batch_size 64 --emb_dim 32 --model 'gin' --pooling 'sum' -c_dim 32 --dataset ${dataset} --seed '[1,2,3]' --dropout 0.0   --contrast 0 -c_in 'raw'  -c_rep 'rep'  --spu_coe 0 --erm --irm_opt 'irm' --num_envs 2 --irm_p 1e-1 --result_name 'irm1' &
CUDA_VISIBLE_DEVICES=2 python main.py --eval_metric 'auc' --r -1 --num_layers 4  --batch_size 64 --emb_dim 32 --model 'gin' --pooling 'sum' -c_dim 32 --dataset ${dataset} --seed '[1,2,3]' --dropout 0.0   --contrast 0 -c_in 'raw'  -c_rep 'rep'  --spu_coe 0 --erm --irm_opt 'irm' --num_envs 2 --irm_p 1 --result_name 'irm2' &
CUDA_VISIBLE_DEVICES=2 python main.py --eval_metric 'auc' --r -1 --num_layers 4  --batch_size 64 --emb_dim 32 --model 'gin' --pooling 'sum' -c_dim 32 --dataset ${dataset} --seed '[1,2,3]' --dropout 0.0   --contrast 0 -c_in 'raw'  -c_rep 'rep'  --spu_coe 0 --erm --irm_opt 'irm' --num_envs 2 --irm_p 1e1  --result_name 'irm3' &
CUDA_VISIBLE_DEVICES=2 python main.py --eval_metric 'auc' --r -1 --num_layers 4  --batch_size 64 --emb_dim 32 --model 'gin' --pooling 'sum' -c_dim 32 --dataset ${dataset} --seed '[1,2,3]' --dropout 0.0   --contrast 0 -c_in 'raw'  -c_rep 'rep'  --spu_coe 0 --erm --irm_opt 'vrex' --num_envs 2 --irm_p 1e-2  --result_name 'vrex' &
wait
CUDA_VISIBLE_DEVICES=2 python main.py --eval_metric 'auc' --r -1 --num_layers 4  --batch_size 64 --emb_dim 32 --model 'gin' --pooling 'sum' -c_dim 32 --dataset ${dataset}  --seed '[1,2,3]' --dropout 0.0   --contrast 0 -c_in 'raw'  -c_rep 'rep'  --spu_coe 0 --erm --irm_opt 'ib-irm' --num_envs 2 --irm_p 1e-2  --result_name 'ibirm' &
CUDA_VISIBLE_DEVICES=2 python main.py --eval_metric 'auc' --r -1 --num_layers 4  --batch_size 64 --emb_dim 32 --model 'gin' --pooling 'sum' -c_dim 32 --dataset ${dataset} --seed '[1,2,3]' --dropout 0.0   --contrast 0 -c_in 'raw'  -c_rep 'rep'  --spu_coe 0 --erm --irm_opt 'eiil' --num_envs 2 --irm_p 1e-2 --result_name 'eiil' &
CUDA_VISIBLE_DEVICES=2 python main.py --eval_metric 'auc' --r -1 --num_layers 4  --batch_size 64 --emb_dim 32 --model 'gin' --pooling 'sum' -c_dim 32 --dataset ${dataset} --seed '[1,2,3]' --dropout 0.0   --contrast 0 -c_in 'raw'  -c_rep 'rep'  --spu_coe 0 --disc 1 --disc_q 0.9 --result_name 'disc0' &
CUDA_VISIBLE_DEVICES=2 python main.py --eval_metric 'auc' --r -1 --num_layers 4  --batch_size 64 --emb_dim 32 --model 'gin' --pooling 'sum' -c_dim 32 --dataset ${dataset} --seed '[1,2,3]' --dropout 0.0   --contrast 0 -c_in 'raw'  -c_rep 'rep'  --spu_coe 0 --disc 1 --disc_q 0.7 --result_name 'disc1' &
CUDA_VISIBLE_DEVICES=2 python main.py --eval_metric 'auc' --r -1 --num_layers 4  --batch_size 64 --emb_dim 32 --model 'gin' --pooling 'sum' -c_dim 32 --dataset ${dataset} --seed '[1,2,3]' --dropout 0.0   --contrast 0 -c_in 'raw'  -c_rep 'rep'  --spu_coe 0 --disc 1 --disc_q 0.5 --result_name 'disc2' &
wait
done;

