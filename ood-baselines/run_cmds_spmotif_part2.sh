
for bb in 0.5001 0.7001;
do

#  CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-5  --num_envs 3 --pretrain 10 -pe 0 | tee -a spmotif_new/1019pspv1_gil5_pe0_spmotif_${bb}.log &
#  CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-3  --num_envs 3 --pretrain 10 -pe 0 | tee -a spmotif_new/1019pspv1_gil3_pe0_spmotif_${bb}.log &
#  CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-1  --num_envs 3 --pretrain 10 -pe 0 | tee -a spmotif_new/1019pspv1_gil1_pe0_spmotif_${bb}.log &
#  CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1  --num_envs 3 --pretrain 10 -pe 0 | tee -a spmotif_new/1019pspv1_gil1_pe0_spmotif_${bb}.log &
# wait
#  CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-5  --num_envs 3 --pretrain 40 | tee -a spmotif_new/1019pspv1_gil5_spmotif_${bb}.log &
#  CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-3  --num_envs 3 --pretrain 40 | tee -a spmotif_new/1019pspv1_gil3_spmotif_${bb}.log &
#  CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-1  --num_envs 3 --pretrain 40 | tee -a spmotif_new/1019pspv1_gil1_spmotif_${bb}.log &
# wait
# CUDA_VISIBLE_DEVICES=1  python3 main_var.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r 0.25 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0. --eiil --irm_p 1  --num_envs 3 -ep 20 | tee -a spmotif_new/1019pspv1_moleOOD_spmotif_${bb}.log &
# CUDA_VISIBLE_DEVICES=1  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'gsat' --model 'gin' --dropout 0.  | tee -a spmotif_new/1019pspv1_gsat_spmotif_${bb}.log &
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 1e-3 --model 'gin' --dropout 0.    --result_name 'grea0' &
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 1e-5 --model 'gin' --dropout 0.    --result_name 'grea1' &
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 1e-1 --model 'gin' --dropout 0.     --result_name 'grea2' &
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 1 --model 'gin' --dropout 0.       --result_name 'grea3' &
wait
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 2 --model 'gin' --dropout 0.      --result_name 'grea4' &
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 -pe 0 --grea 10 --model 'gin' --dropout 0.     --result_name 'grea5' &
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 5 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'gsat' --model 'gin' --dropout 0.    --result_name 'gsat' &
# wait
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-5  --num_envs 3 --pretrain 10 -pe 10    --result_name 'gil0' &
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-3  --num_envs 3 --pretrain 10 -pe 10    --result_name 'gil1' &
wait
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1e-1  --num_envs 3 --pretrain 10 -pe 10   --result_name 'gil2' &
 CUDA_VISIBLE_DEVICES=2  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --gil 1  --num_envs 3 --pretrain 10 -pe 10   --result_name 'gil3' &
#  CUDA_VISIBLE_DEVICES=2  python3 main_var.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r 0.25 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0. --eiil --irm_p 1  --num_envs 3 -ep 20 | tee -a spmotif_new/1019pspv1_moleOOD_spmotif_${bb}.log &
wait

done;

