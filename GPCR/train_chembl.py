# -*- coding: utf-8 -*-
"""
@Time:Created on 2023/10/2 
@author: Yoyo Wu
Revision DESSML benchmark 
Train TransformerCPI with Chembl
"""
from email import parser
import torch
import numpy as np
import random
import os
import time
from model import *
import timeit
import json
import argparse
from utils import load_data_map, ProteinCompoundDataset, collate_fn, set_seed
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler 
import time
import wandb



if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='hmdb_test_tiny', help='run name')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--train_data', type=str, default='/raid/home/yoyowu/TransformerCPI/dataset/chembl/all_Chembl29.tsv', help='training data path')
    parser.add_argument('--unl_data', type=str, default='/raid/home/yoyowu/TransformerCPI/dataset/hmdb/hmdb_target_sample.tsv', help='unlabeled data path')
    parser.add_argument('--val_data', type=str, default='/raid/home/yoyowu/TransformerCPI/dataset/hmdb/Feb_13_23_dev_test/dev_47.tsv', help='validation data path')
    parser.add_argument('--test_data', type=str, default='/raid/home/yoyowu/TransformerCPI/dataset/hmdb/Feb_13_23_dev_test/test_47.tsv', help='test data path')
    parser.add_argument('--iteration', type=int, default=100, help='number of iteration')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--chembl_smiles_path', type=str, default='/raid/home/yoyowu/TransformerCPI/dataset/chembl/chembl2smiles.tsv', help='smiles path')
    parser.add_argument('--chembl_protein_path', type=str, default='/raid/home/yoyowu/TransformerCPI/dataset/chembl/10022023_chembl_prot_seq.json', help='protein path')
    parser.add_argument('--chembl_uniprot_mapping_path', type=str, default='/raid/home/yoyowu/TransformerCPI/dataset/chembl/chembl_uniprot_mapping.json', help='chembl_uniprot_mapping path')
    parser.add_argument('--hmdb_smiles_path', type=str, default='/raid/home/yoyowu/TransformerCPI/dataset/hmdb/compounds_smiles.json', help='smiles path')
    parser.add_argument('--hmdb_protein_path', type=str, default='/raid/home/yoyowu/TransformerCPI/dataset/hmdb/10042023_hmdb_prot_seq.json', help='protein path')
    parser.add_argument("--hmdb_uniprot_mapping_path", type=str, default='/raid/home/yoyowu/TransformerCPI/dataset/hmdb/hmdb_uniprot_mapping.json', help='hmdb_uniprot_mapping path')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--model_save_path', type=str, default='/raid/home/yoyowu/TransformerCPI/saved_model', help='save model path')
    parser.add_argument('--wandb', type=bool, default=True, help='use wandb or not')
    parser.add_argument('--amp', type=bool, default=True, help='use mix precision training or not')
    parser.add_argument('--num_workers',type=int, default=40)
    parser.add_argument('--from_checkpoint', type=str, default=None, help='load model from checkpoint')

    args = parser.parse_args()
    # ------------pass the fixe parameters-------------
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    kernel_size = 7
    shuffle_dataset = True
    num_workers = args.num_workers  # Adjust based on your system's capabilities
    
    if args.wandb:

        wandb.init(project="transformerCPI", name=args.run_name)
        wandb.config.update(args)

    device = torch.device('cuda:{}'.format(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
    set_seed(args.seed)
    with open(args.train_data, "r") as f:
        lines = f.readlines()
        tr_data = [line.strip().split('\t') for line in lines]
    with open(args.unl_data, "r") as f:
        lines = f.readlines()
        unl_data = [line.strip().split('\t') for line in lines]
    with open(args.val_data, "r") as f:
        lines = f.readlines()
        val_data = [line.strip().split('\t') for line in lines]
    with open(args.test_data, "r") as f:
        lines = f.readlines()
        test_data = [line.strip().split('\t') for line in lines]

    chembl_smiles_mapping, chembl_protein_seq_mapping, chembl_uniprot_mapping = load_data_map(args.chembl_smiles_path, args.chembl_protein_path, args.chembl_uniprot_mapping_path)
    hmdb_smiles_mapping, hmdb_protein_seq_mapping, hmdb_uniprot_mapping = load_data_map(args.hmdb_smiles_path, args.hmdb_protein_path, args.hmdb_uniprot_mapping_path)

    # Initialize dataset with the Chembl to Uniprot mapping
    tr_dataset = ProteinCompoundDataset(tr_data,  chembl_smiles_mapping, chembl_protein_seq_mapping, chembl_uniprot_mapping)
    unl_dataset = ProteinCompoundDataset(unl_data, hmdb_smiles_mapping, hmdb_protein_seq_mapping, hmdb_uniprot_mapping, unlabeled=True)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch, shuffle=shuffle_dataset, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
    val_dataset = ProteinCompoundDataset(val_data, hmdb_smiles_mapping, hmdb_protein_seq_mapping, hmdb_uniprot_mapping)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=shuffle_dataset, num_workers=num_workers, drop_last=False, collate_fn=collate_fn)
    test_dataset = ProteinCompoundDataset(test_data, hmdb_smiles_mapping, hmdb_protein_seq_mapping, hmdb_uniprot_mapping)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=shuffle_dataset, num_workers=num_workers, drop_last=False, collate_fn=collate_fn)

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)

    model.to(device)
    #model = nn.DataParallel(model)
    trainer = Trainer(model, lr, weight_decay, args.batch, args.amp, args.from_checkpoint)
    tester = Tester(model)
    # Early Stopping Variables
    best_auc_val = 0
    best_auc_test = 0
    best_val_epoch = -1
    best_test_epoch = -1

    epochs_no_improve = 0
    epoch_info = []
    overall_start_time = time.time()  # Start the overall timer

    print("Start training...")
    print("The current configuration is: {}".format(args))
    for epoch in range(1, args.iteration+1):
        start_time = time.time()  # Start the timer
        loss_train = trainer.train(tr_dataloader, device)
        #Test on training data (if needed)
        AUC_tr, PRC_tr = tester.test(tr_dataloader, device)
        # Test on validation data
        AUC_val, PRC_val = tester.test(val_dataloader, device)  # You'll need to define 'validation_dataloader' similar to 'dataloader'
        # Test on test data
        AUC_test, PRC_test = tester.test(test_dataloader, device)
        if args.wandb:
            wandb.log({
                "loss_train": loss_train,
                "AUC_tr": AUC_tr,
                "PRC_tr": PRC_tr,
                "AUC_val": AUC_val,
                "PRC_val": PRC_val,
                "AUC_test": AUC_test,
                "PRC_test": PRC_test
            })
        # If the validation AUC is greater than our current best
        if AUC_val > best_auc_val:
            best_auc_val = AUC_val
            best_val_epoch = epoch
            epochs_no_improve = 0
            #Save the model if desired
            if args.model_save_path is not None:
                torch.save(model.state_dict(), str(args.model_save_path)+'/'+str(args.run_name)+'.pt')
                print("Model saved at {}. Best AUC: {}".format(args.model_save_path, best_auc_val))
        
        else:
            epochs_no_improve += 1
            if epochs_no_improve == args.patience:
                print("Early stopping!")
                break
    # Store results for the epoch
        if AUC_test > best_auc_test:
            best_auc_test = AUC_test
            best_test_epoch = epoch

        epoch_results = {
            "epoch": epoch,
            "loss_train": loss_train,
            "AUC_tr": AUC_tr,
            "PRC_tr": PRC_tr,
            "AUC_val": AUC_val,
            "PRC_val": PRC_val,
            "AUC_test": AUC_test,
            "PRC_test": PRC_test
        }
        epoch_info.append(epoch_results)

 

        end_time = time.time()  # End the timer

        elapsed_time = end_time - start_time  # Calculate the elapsed time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Epoch {epoch} finished. Time taken: {minutes}m {seconds}s.")
# Write results to JSON file
with open("output/{}_results.json".format(args.run_name), "w") as f:
    json.dump(epoch_info, f)
print(f"Best validation AUC: {best_auc_val} at epoch {best_val_epoch}")
print(f"Test AUC at that epoch: {epoch_info[best_val_epoch - 1]['AUC_test']}")
print(f"Best test AUC: {best_auc_test} at epoch {best_test_epoch}")
overall_end_time = time.time()  # End the overall timer
total_elapsed_time = overall_end_time - overall_start_time  # Calculate the total elapsed time
total_minutes = int(total_elapsed_time // 60)
total_seconds = int(total_elapsed_time % 60)
print(f"Total training time: {total_minutes}m {total_seconds}s.")