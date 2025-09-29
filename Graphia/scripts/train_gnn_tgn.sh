# train lp reward model
epochs=200
save_root=Graphia
data_root=Graphia/data

python -m Graphia.train_link_prediction --data_name 8days_dytag_small_text_en --pred_ratio 0.15 --time_window 86400 --bwr 1980 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ >> gnn_tgn/log_gnn_lp.txt 2>&1 &

nohup python -m Graphia.train_edge_classification --data_name 8days_dytag_small_text_en --pred_ratio 0.15 --time_window 86400 --bwr 1980 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/ec_models/ >> gnn_tgn/log_gnn_ec.txt 2>&1 &

nohup python -m Graphia.train_link_prediction --data_name weibo_daily --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ >> gnn_tgn/log_gnn_lp_weibo_daily.txt 2>&1 &

nohup python -m Graphia.train_edge_classification --data_name weibo_daily --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/ec_models/ >> gnn_tgn/log_gnn_ec_weibo_daily.txt 2>&1 &

nohup python -m Graphia.train_link_prediction --data_name weibo_tech --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ >> gnn_tgn/log_gnn_lp_weibo_tech.txt 2>&1 &

nohup python -m Graphia.train_edge_classification --data_name weibo_tech --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/ec_models/ >> gnn_tgn/log_gnn_ec_weibo_tech.txt 2>&1 &



nohup python -m Graphia.evaluate_node_retrieval --data_name 8days_dytag_small_text_en --pred_ratio 0.15 --time_window 86400 --bwr 1980 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ >> gnn_tgn/log_gnn_nr_8days.txt 2>&1 &

nohup python -m Graphia.evaluate_node_retrieval --data_name weibo_daily --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ >> gnn_tgn/log_gnn_nr_weibo_daily.txt 2>&1 &

nohup python -m Graphia.evaluate_node_retrieval --data_name weibo_tech --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/lp_models/ >> gnn_tgn/log_gnn_nr_weibo_tech.txt 2>&1 &
