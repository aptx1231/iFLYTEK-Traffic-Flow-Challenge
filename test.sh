# 数据处理
# python data_process.py

# 加载模型，生成预测结果
for id in 82400 42186 8906 60799 74660 23462 61922 63374 60998 71010
do
  python run_model.py --model AGCRN --gpu_id 0 --rnn_units 128 --embed_dim 30 --cheb_order 4 --batch_size 16 --path ../user_data/model_data/$id/AGCRN_xunfei.m --train False --exp_id $id
done

# 模型融合
python fusion.py
