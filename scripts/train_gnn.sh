# train lp reward model
epochs=200
save_root=/data/oss_bucket_0/jjr/LLMGGen
data_root=/data/oss_bucket_0/jjr/LLMGGen

# nohup python LLMGGen/train_link_prediction.py --data_name 8days_dytag_small_text_en --pred_ratio 0.15 --time_window 86400 --bwr 1980 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name GraphMixer --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ >> log_gnn_lp.txt 2>&1 &

# nohup python LLMGGen/train_edge_classification.py --data_name 8days_dytag_small_text_en --pred_ratio 0.15 --time_window 86400 --bwr 1980 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name GraphMixer --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/ec_models/ >> log_gnn_ec.txt 2>&1 &

# nohup python -m LLMGGen.train_link_prediction --data_name weibo_daily --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name GraphMixer --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ >> log_gnn_lp_weibo_daily.txt 2>&1 &

# nohup python -m LLMGGen.train_edge_classification --data_name weibo_daily --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name GraphMixer --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/ec_models/ >> log_gnn_ec_weibo_daily.txt 2>&1 &

# nohup python -m LLMGGen.train_link_prediction --data_name weibo_tech --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name GraphMixer --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ >> log_gnn_lp_weibo_tech.txt 2>&1 &

# nohup python -m LLMGGen.train_edge_classification --data_name weibo_tech --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name GraphMixer --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/ec_models/ >> log_gnn_ec_weibo_tech.txt 2>&1 &


# nohup python -m LLMGGen.train_link_prediction --data_name imdb --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name GraphMixer --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ >> log_gnn_lp_imdb.txt 2>&1 &

# nohup python -m LLMGGen.train_edge_classification --data_name imdb --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name GraphMixer --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/ec_models/ >> log_gnn_ec_imdb.txt 2>&1 &

python -m LLMGGen.evaluate_node_retrieval --data_name 8days_dytag_small_text_en --pred_ratio 0.15 --time_window 86400 --bwr 1980 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name GraphMixer --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ --negative_sample_strategy  historical
