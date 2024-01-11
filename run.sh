run_name=1012_chembl_hmdb_1k_b_bs_8_split_27_dessml
python  GPCR/train_chembl.py --run_name $run_name  --iteration 100 --device 3 --wandb True \
--from_checkpoint  /raid/home/yoyowu/TransformerCPI/saved_model/1006_chembl_hmdb_1k_b_split_27_continue.pt \
> ./logs/$run_name.log 2>&1 &