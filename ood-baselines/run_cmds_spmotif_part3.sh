
for bb in 0.5001 0.7001;
do
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'ciga' --model 'gin' --dropout 0.    --result_name 'ciga0' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 2 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'ciga' --model 'gin' --dropout 0.    --result_name 'ciga1' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 8 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'ciga' --model 'gin' --dropout 0.    --result_name 'ciga2' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 16 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'ciga' --model 'gin' --dropout 0.    --result_name 'ciga3' &
wait
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 0 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'gala' --model 'gin' --dropout 0.    --result_name 'gala0' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 2 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'gala' --model 'gin' --dropout 0.    --result_name 'gala1' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 8 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'gala' --model 'gin' --dropout 0.    --result_name 'gala2' &
CUDA_VISIBLE_DEVICES=3  python3 main.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3]' --num_layers 5 --pretrain 10 --batch_size 32 --dataset 'SPMotif' --bias ${bb} --r -1 --contrast 16 --spu_coe 0 --irm_p 1 --r 0.7 --ginv_opt 'gala' --model 'gin' --dropout 0.    --result_name 'gala3' &
wait
done;

