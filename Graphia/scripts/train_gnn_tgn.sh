# train lp reward model
epochs=200
save_root=Graphia
data_root=Graphia/data


python -m Graphia.train_edge_classification --data_name 8days_dytag_small_text_en --pred_ratio 0.15 --time_window 86400 --bwr 1980 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/ec_models/ 

python -m Graphia.train_edge_classification --data_name weibo_daily --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/ec_models/

python -m Graphia.train_edge_classification --data_name weibo_tech --pred_ratio 0.15 --time_window 86400 --bwr 2048 --use_feature bert --cm_order --num_runs 1 --gpu 0 --num_epochs $epochs --patience 5 --model_name TGN --num_layers 2 --num_neighbors 20 --dropout 0.1 --sample_neighbor_strategy recent --load_best_configs --data_root ${data_root}/ --save_root ${save_root}/ec_models/ 
