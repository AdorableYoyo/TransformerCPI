# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/17 8:36
@author: LiFan Chen
@Filename: model.py
@Software: PyCharm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from Radam import *
from lookahead import Lookahead
from torch.cuda.amp import autocast, GradScaler
from itertools import islice


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        #protein = protein + self.pos_embedding(pos)
        #protein = [batch size, protein len,protein_dim]
        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]
        conved = self.ln(conved)
        return conved



class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.gn = nn.GroupNorm(8, 256)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)

        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src,trg_mask,src_mask)

        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg, dim=2)
        # norm = [batch size,compound len]
        norm = F.softmax(norm, dim=1)
        # norm = [batch size,compound len]
        # trg = torch.squeeze(trg,dim=0)
        # norm = torch.squeeze(norm,dim=0)
        sum = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j, ]
                v = v * norm[i, j]
                sum[i, ] += v
        # sum = [batch size,hid_dim]
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)
        return label


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=34):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()
        self.Loss= nn.CrossEntropyLoss()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def gcn(self, input, adj):
        # input =[batch,num_node, atom_dim]
        # adj = [batch,num_node, num_node]
        support = torch.matmul(input, self.weight)
        # support =[batch,num_node,atom_dim]
        output = torch.bmm(adj, support)
        # output = [batch,num_node,atom_dim]
        return output

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)  # batch size
        atom_num = atom_num.long()
        protein_num = protein_num.long()
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask


    def forward(self, data):
        if len(data) == 5:
            compound, adj, protein, atom_num,  protein_num = data
            correct_interaction = None
        elif len(data) == 6:
            compound, adj, protein, correct_interaction, atom_num, protein_num = data
        else:
            raise ValueError("data length is not correct")
        
        # compound = [batch,atom_num, atom_dim]
        # adj = [batch,atom_num, atom_num]
        # protein = [batch,protein len, 100]

        compound_max_len = compound.shape[1]
        protein_max_len = protein.shape[1]
        compound_mask, protein_mask = self.make_masks(atom_num, protein_num, compound_max_len, protein_max_len)
        compound = self.gcn(compound, adj)
        # compound = torch.unsqueeze(compound, dim=0)
        # compound = [batch size=1 ,atom_num, atom_dim]

        # protein = torch.unsqueeze(protein, dim=0)
        # protein =[ batch size=1,protein len, protein_dim]
        enc_src = self.encoder(protein)
        # enc_src = [batch size, protein len, hid dim]

        predicted_interaction = self.decoder(compound, enc_src, compound_mask, protein_mask)
        # out = [batch size, 2]
        # out = torch.squeeze(out, dim=0)
        if correct_interaction is not None:
            correct_interaction = correct_interaction.long()
        ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
        predicted_labels = np.argmax(ys, axis=1)
        predicted_scores = ys[:, 1]
        return predicted_interaction, correct_interaction, predicted_labels, predicted_scores
        # if pseudo_label is None:
        #     loss = self.Loss(predicted_interaction, correct_interaction)
        # else:
        #     loss = nn.BCEWithLogitsLoss()(predicted_interaction, correct_interaction)
        # return loss, correct_interaction, predicted_interaction, predicted_labels, predicted_scores


    # def __call__(self, data, train=True):

    #     compound, adj, protein, correct_interaction ,atom_num,protein_num = data
    #     # compound = compound.to(self.device)
    #     # adj = adj.to(self.device)
    #     # protein = protein.to(self.device)
    #     # correct_interaction = correct_interaction.to(self.device)
    #     Loss = nn.CrossEntropyLoss()

    #     if train:
    #         predicted_interaction = self.forward(compound, adj, protein,atom_num,protein_num)
    #         correct_interaction = correct_interaction.long()
    #         loss = Loss(predicted_interaction, correct_interaction)
    #         return loss

    #     else:
    #         #compound = compound.unsqueeze(0)
    #         #adj = adj.unsqueeze(0)
    #         #protein = protein.unsqueeze(0)
    #         #correct_interaction = correct_interaction.unsqueeze(0)
    #         predicted_interaction = self.forward(compound, adj, protein,atom_num,protein_num)
    #         correct_labels = correct_interaction.to('cpu').data.numpy()
    #         ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
    #         predicted_labels = np.argmax(ys, axis=1)
    #         predicted_scores = ys[:, 1]
    #         return correct_labels, predicted_labels, predicted_scores


def pack(atoms, adjs, proteins, labels, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        i += 1
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return (atoms_new, adjs_new, proteins_new, labels_new, atom_num, protein_num)
def to_cuda(data, device, cuda_available=True):
    if len(data) == 5:
        compound, adj, protein, atom_num,  protein_num = data
        correct_interaction = None
    else:
        compound, adj, protein, correct_interaction, atom_num, protein_num = data

    # Put input to cuda
    if cuda_available:
        compound = compound.to(device)
        adj = adj.to(device)
        protein = protein.to(device)
        atom_num = atom_num.to(device)
        protein_num = protein_num.to(device)
        if correct_interaction is not None:
            correct_interaction = correct_interaction.to(device)

    return compound, adj, protein, correct_interaction, atom_num, protein_num

class Trainer(object):
    def __init__(self, teacher_model, student_model, lr, weight_decay, amp, checkpoint=None):
        self.teacher_model = teacher_model
        self.student_model = student_model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []
        self.amp = amp
        self.temperature = 0.7

        for p in self.teacher_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if checkpoint is not None:
            #self.model.load_state_dict(checkpoint)
            self.teacher_model.load_state_dict(torch.load(checkpoint))
            print('load model from checkpoint {}'.format(checkpoint))

        # Splitting the parameters for weight decay differentiation
        t_weight_p, t_bias_p = self._split_parameters(self.teacher_model)
        s_weight_p, s_bias_p = self._split_parameters(self.student_model)
        
        # Optimizers
        self.t_optimizer = self._init_optimizer(t_weight_p + t_bias_p + s_weight_p + s_bias_p, lr, weight_decay)
        self.s_optimizer = self._init_optimizer(s_weight_p + s_bias_p, lr, weight_decay)

    def _split_parameters(self, model):
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        return weight_p, bias_p
    
    def _init_optimizer(self, parameters, lr, weight_decay):
        weight_p, bias_p = [], []
        for param in parameters:
            if param.dim() > 1:
                weight_p.append(param)
            else:
                bias_p.append(param)

        optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)
        return optimizer


        # for name, p in self.teacher_model.named_parameters():
        #     if 'bias' in name:
        #         bias_p += [p]
        #     else:
        #         weight_p += [p]
        # # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        # self.optimizer_inner = RAdam(
        #     [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        # self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
       
 

    def train(self, dataloader, unl_dataloader, device):
        s_scaler = GradScaler()
        t_scaler = GradScaler()
        self.teacher_model.train()
        self.student_model.train()

        s_total_loss = 0.0
        t_total_loss = 0.0
        num_batches = 0

        #Create an iterator for the unlabeled dataloader
        unl_iterator = iter(unl_dataloader)
        #for i, data_pack in enumerate(dataloader):
        # sample few batches
        for i, data_pack in islice(enumerate(dataloader), 0, 1000):
            data_pack = to_cuda(data_pack, device=device)
            try :
                # try to get a batch from the unlabeled dataloader
                unl_data= next(unl_iterator)
            except StopIteration:
                unl_iterator = iter(unl_dataloader)
                unl_data = next(unl_iterator)
            unl_data = to_cuda(unl_data, device=device)

            self.t_optimizer.zero_grad()
            self.s_optimizer.zero_grad()
            if self.amp:
                with autocast():  # Enable mixed precision for the forward pass
                    t_logits, _, _, _= self.teacher_model(unl_data)
                    pseudo_labels = torch.softmax(t_logits.detach() / self.temperature, dim=-1) # apply temperature on the soft lbs
                    s_logits, labels, _, _ = self.student_model(data_pack)
                    s_loss = nn.BCEWithLogitsLoss()(s_logits, pseudo_labels)

                    #loss, correct_interaction, predicted_labels, predicted_scores
            #Backward pass and optimization using the scaler
                s_scaler.scale(s_loss).backward()
                s_scaler.step(self.s_optimizer)
                s_scaler.update()
                with autocast():
                    s_logits_new, _, _, _ = self.student_model(data_pack)
                    t_loss = torch.nn.CrossEntropyLoss()(s_logits_new, labels)
                t_scaler.scale(t_loss).backward()
                t_scaler.step(self.t_optimizer)
                t_scaler.update()
            else:
                print("no amp")
                # loss, _, _, _ = self.model(data_pack)
                # loss.backward()
                # self.optimizer.step()
            s_total_loss += s_loss.item()
            t_total_loss += t_loss.item()
            num_batches += 1
        s_average_loss = s_total_loss / num_batches
        t_average_loss = t_total_loss / num_batches
        return s_average_loss , t_average_loss   
         


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataloader, device, threshold=7., plot=False):
        self.model.eval()

        #T, S, Y = torch.Tensor(), torch.Tensor(), torch.Tensor()
        T, S, Y = [], [], []
        with torch.no_grad():
            for i, data_pack in islice(enumerate(dataloader), 0, 500):
            #for i, data_pack in enumerate(dataloader):
                data_pack = to_cuda(data_pack, device=device)

                _, correct_interaction, predicted_labels, predicted_scores = self.model(data_pack)

                T.extend(correct_interaction.cpu().detach())
                Y.extend(list(predicted_labels))
                S.extend(list(predicted_scores))
                #Y = torch.cat((S, predicted_labels.cpu().detach()))

        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        return AUC, PRC


    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.module.state_dict(), filename + ".state_dict")
        torch.save(model, filename + ".entire_model")