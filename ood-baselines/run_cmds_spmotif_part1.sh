for bb in 0.5001 0.7001;
do
CUDA_VISIBLE_DEVICES=3   python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 50 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --erm --irm_opt 'irm' --num_envs 2 --irm_p 1e-2 --spu_coe 0 --model 'gin' --dropout 0.   --result_name 'irm0' &
CUDA_VISIBLE_DEVICES=3   python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 50 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --erm --irm_opt 'irm' --num_envs 2 --irm_p 1e-1 --spu_coe 0 --model 'gin' --dropout 0.   --result_name 'irm1' &
CUDA_VISIBLE_DEVICES=3   python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 50 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --erm --irm_opt 'irm' --num_envs 2 --irm_p 1 --spu_coe 0 --model 'gin' --dropout 0.      --result_name 'irm2' &
# CUDA_VISIBLE_DEVICES=3   python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 3 --pretrain 50 --batch_size 32 --dataset 'spmotif' --bias ${bb} --erm --irm_opt 'irm' --num_envs 2 --irm_p 1e1 --spu_coe 0 --model 'gin' --dropout 0.  | tee -a spmotif_layer5/1019pspv1_irm_spmotif_${bb}.log &
wait
CUDA_VISIBLE_DEVICES=3   python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 50 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --erm --irm_opt 'irm' --num_envs 2 --irm_p 1e2 --spu_coe 0 --model 'gin' --dropout 0.      --result_name 'irm3' &
CUDA_VISIBLE_DEVICES=3   python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 50 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --erm --irm_opt 'vrex' --num_envs 2 --irm_p 1e-2 --spu_coe 0 --model 'gin' --dropout 0.    --result_name 'vrex' &
CUDA_VISIBLE_DEVICES=3   python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 50 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --erm --irm_opt 'ib-irm' --num_envs 2 --irm_p 1e-2 --spu_coe 0 --model 'gin' --dropout 0.   --result_name 'ibirm' &
CUDA_VISIBLE_DEVICES=3   python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 50 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --erm --irm_opt 'eiil' --num_envs 2 --irm_p 1e-2 --spu_coe 0 --model 'gin' --dropout 0.    --result_name 'eiil' &
wait
CUDA_VISIBLE_DEVICES=3   python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 50 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --r 0.25 --disc 1 --disc_q 0.9   --result_name 'disc0' &
CUDA_VISIBLE_DEVICES=3   python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 50 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --r 0.25 --disc 1 --disc_q 0.7   --result_name 'disc1' &
CUDA_VISIBLE_DEVICES=3   python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 50 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --model 'gin' --dropout 0.  --r 0.25 --disc 1 --disc_q 0.5   --result_name 'disc2' &
wait
done;


# for bb in 0.6;
# do
# CUDA_VISIBLE_DEVICES=0   python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 3 --pretrain 50 --batch_size 32 --dataset 'spmotif' --bias ${bb} --erm --irm_opt 'vrex' --num_envs 2 --irm_p 1 --spu_coe 0 --model 'gin' --dropout 0.   | tee -a spmotif_layer5/1019pspv1_irm2_spmotif_${bb}.log &
# CUDA_VISIBLE_DEVICES=3   python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 3 --pretrain 50 --batch_size 32 --dataset 'spmotif' --bias ${bb} --erm --irm_opt 'ib-irm' --num_envs 2 --irm_p 1 --spu_coe 0 --model 'gin' --dropout 0. | tee -a spmotif_layer5/1019pspv1_irm2_spmotif_${bb}.log &
# CUDA_VISIBLE_DEVICES=3   python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 3 --pretrain 50 --batch_size 32 --dataset 'spmotif' --bias ${bb} --erm --irm_opt 'eiil' --num_envs 2 --irm_p 1 --spu_coe 0 --model 'gin' --dropout 0.   | tee -a spmotif_layer5/1019pspv1_irm2_spmotif_${bb}.log &
# wait
# done;



