import torch
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForPreTraining, AlbertForPreTraining, DebertaV2PreTrainedModel, DebertaV2ForMaskedLM
from dataset import pad_or_truncate
import matplotlib.pyplot as plt
import seaborn as sns

# pooling
def extract_entity(sequence_output, e_mask):
    # print(e_mask.size())   # [32, 128]
    extended_e_mask = e_mask.unsqueeze(-1)
    # print(extended_e_mask.size()) # [32, 128, 1]
    extended_e_mask = extended_e_mask.float() * sequence_output
    extended_e_mask, _ = extended_e_mask.max(dim=-2)
    # print(extended_e_mask.size()) # [32, 768]
    # extended_e_mask = torch.stack([sequence_output[i,j,:] for i,j in enumerate(e_mask)])
    # print(extended_e_mask)
    return extended_e_mask.float()


class EMMA(nn.Module):  
    def __init__(self, pretrain_model_name_or_path, add_auto_match=True, max_seq_len=128, k=3):
        super(EMMA, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrain_model_name_or_path)
        self.bert = AutoModel.from_pretrained(pretrain_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name_or_path)
        self.max_seq_len = max_seq_len
        self.k = k
        self.cos = nn.CosineSimilarity(dim=-1)
        # self.alpha = nn.Parameter(torch.tensor(0.3))
        self.add_auto_match = add_auto_match
        
        self.encoder_fc1 = nn.Linear(3*self.config.hidden_size, self.config.hidden_size)
        self.encoder_fc_mu = nn.Linear(self.config.hidden_size, 128)     # 输出均值 μ
        self.encoder_fc_logvar = nn.Linear(self.config.hidden_size, 128) # 输出对数方差 log(σ^2)

        self.decoder_fc1 = nn.Linear(128 + 2*self.config.hidden_size, 512)
        self.decoder_fc_out = nn.Linear(512, self.config.hidden_size) # 输出拼接的 [h_emb_rec, t_emb_rec]
        # if add_auto_match:
        #     self.des_weights1 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
        #     self.des_bias1 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
        #     self.des_weights2 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
        #     self.des_bias2 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
        #     # self.sen_att_cls = MultiHeadAttention(self.config.hidden_size, 0.1, 8)
        #     # self.sen_att_head = MultiHeadAttention(self.config.hidden_size, 0.1, 8)
        #     # self.sen_att_tail = MultiHeadAttention(self.config.hidden_size, 0.1, 8)
        #     # self.sen_att = MultiHeadAttention(self.config.hidden_size, 0.1, 8)
        # self.project_head = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        # self.classify1 = nn.Linear(4*self.k*self.config.hidden_size, 4*self.config.hidden_size)
        # self.classify2 = nn.Linear(4*self.config.hidden_size, self.config.hidden_size)
        # self.classify3 = nn.Linear(self.config.hidden_size, self.k)

        if add_auto_match:
            self.des_weights1 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
            self.des_bias1 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
            self.des_weights2 = nn.Parameter(torch.ones(self.config.hidden_size, 1))
            self.des_bias2 = nn.Parameter(torch.zeros(max_seq_len - 1, 1))
        
        # self.classify = Classify_model(pretrain_model_name_or_path, max_seq_len, k)
        

        self.des_vectors = None
    
    
    # def forward(self, sen_input_ids, sen_att_masks, des_input_ids, des_att_masks, sen_e1_pos, sen_e2_pos, sen_e1_pos_end, sen_e2_pos_end):
    def forward(self, sen_input_ids, sen_att_masks, des_input_ids=None, des_att_masks=None, \
                marked_e1=None, marked_e2=None, mark_head=None, mark_tail=None, head_range=None, tail_range=None):
        '''
        sen_input_ids: [bs, max_seq_length]
        sen_att_masks: [bs, max_seq_length]
        des_input_ids: [bs, max_seq_length]
        des_att_masks: [bs, max_seq_length]
        marked_e1: [bs, max_seq_length] 
        marked_e2: [bs, max_seq_length]
        mark_head: [bs, max_seq_length]
        mark_tail: [bs, max_seq_length]
        '''
        batch_size = sen_input_ids.size(0)
        device = sen_input_ids.device
        
        # sentence = [cls] + ... + [E1] + E1 + [E1/] + ... + [E2] + E2 + [E2/] + .....
        # label_description = [cls] + .....

        sen_outputs = self.bert(
            input_ids=sen_input_ids,
            attention_mask=sen_att_masks,
        )
        sen_output = sen_outputs.last_hidden_state
        sen_vec, sen_e1, sen_e2 = self.get_sen_vec(sen_output, marked_e1, marked_e2)
        # sen_local_feature = self.get_sen_entity_between_vec(sen_output, marked_e1, marked_e2)

        if des_input_ids != None:
            des_outputs = self.bert(
                input_ids=des_input_ids,
                attention_mask=des_att_masks,
            )
            des_output = des_outputs.last_hidden_state
            des_vec = self.get_des_vec(des_output)
            des_vec_feature = torch.mean(des_output,dim=1)
        # [cls] xxx xxx xxx [E1] head_entity [E1/] xxx xxx [E2] tail_entity [E2/] xxx xxx xxx
        # e1_h = extract_entity(sen_output, marked_e1) # [E1]
        # e2_h = extract_entity(sen_output, marked_e2) # [E2]
        # head_entity = extract_entity(sen_output, mark_head) # head_entity
        # tail_entity = extract_entity(sen_output, mark_tail) # tail_entity
        
        # train
        if des_input_ids != None:

            #生成“造句”特征
            x = torch.cat([sen_e1*sen_e2, sen_e1, sen_e2], dim=-1)
            hidden = F.relu(self.encoder_fc1(x))
            mu = self.encoder_fc_mu(hidden)
            logvar = self.encoder_fc_logvar(hidden)

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) # 从 N(0, I) 采样
            z = mu + eps * std

            y = torch.cat([z, sen_e1, sen_e2], dim=-1)
            hidden = F.relu(self.decoder_fc1(y))
            reconstructed_emb_concat = self.decoder_fc_out(hidden) # sigmoid 或 tanh 如果原始emb被归一化

            recon_sen = torch.cat([reconstructed_emb_concat,sen_e1, sen_e2],dim=-1)

            # 双塔loss
            cos_sim = self.cos(sen_vec.unsqueeze(1), des_vec.unsqueeze(0)) # [bs, 1, 768]   [1, bs, 768]

            con_labels = torch.arange(batch_size).long().to(device)
            loss_fct = nn.CrossEntropyLoss() 

            hb_loss_fn = nn.HuberLoss(delta=1.0)

            kl_loss_elements = mu.pow(2) + logvar.exp() - logvar - 1
            kl_loss_per_sample = 0.5 * torch.sum(kl_loss_elements, dim=1) # Sum over latent dimensions

            # To get a single scalar loss value for the batch, you often average over the batch:
            L_kl = torch.mean(kl_loss_per_sample) # Mean over batch dimension
            L_hb = hb_loss_fn(reconstructed_emb_concat, des_vec_feature)
        
            cl_loss = loss_fct(cos_sim / 0.02, con_labels)


            cls_cos_sim = self.cos(recon_sen.unsqueeze(1), des_vec.unsqueeze(0))
            cls_loss = loss_fct(cls_cos_sim / 0.02, con_labels)


            loss = cl_loss + 0.1 * L_hb + 0.05 * L_kl + cls_loss
            

            # detach: classify模型的梯度不会回传到EMMA，也就是两个模型互不影响
            # sen_e1_vec, sen_e2_vec, rec_emb, des_vec_feature, target_idx_arr, vec_idx_arr = \
            #     self.build_classifaction_input(sen_e1, sen_e2, des_vec_feature, training=True, reconstructed_emb_concat=reconstructed_emb_concat)
            
            # class_feature = torch.cat([sen_e1_vec, sen_e2_vec, rec_emb, des_vec_feature],dim=-1)
            # cls_feature = class_feature.reshape(batch_size, self.k*4*self.config.hidden_size)
            # x1 = self.classify1(cls_feature)
            # x2 = self.classify2(torch.relu(x1))
            # logits = self.classify3(torch.relu(x2))

            # classify_loss = loss_fct(logits, target_idx_arr)
            
            return loss
        else:
            # cos_sim_ctx = self.cos(context_proj.unsqueeze(1), eval_des_features[:,0,:].unsqueeze(0)) # [32, 1, 768] # [1, m, 768]
            # cos_sim_e1 = self.cos(e1_proj.unsqueeze(1), eval_des_features[:,1,:].unsqueeze(0)) # expand : [32, m, 768] # [32, m, 768]
            # cos_sim_e2 = self.cos(e2_proj.unsqueeze(1), eval_des_features[:,2,:].unsqueeze(0)) # result : [32, m]
            # integrated_sim = self.alpha*(cos_sim_e1 + cos_sim_e2) + (1-self.alpha)*cos_sim_ctx
            # cos_sim = self.cos(sen_vec.unsqueeze(1), self.des_vectors.unsqueeze(0)) # [bs, 1, 768]   [1, m, 768] -> [bs, m]
            cos_sim = self.cos(sen_vec.unsqueeze(1), self.des_vectors.unsqueeze(0))
            # recon_sim = self.cos(reco_sen_vec.unsqueeze(1), self.des_vectors.unsqueeze(0))
            max_sim, max_sim_idx = torch.max(cos_sim, dim=1)  # 获取相似度最大的一列
            
            # 分类
            # top_k_values, top_k_indices = torch.topk(cos_sim, self.k, dim=1)
            # # print(self.des_vectors_feature.shape)
            # sen_e1_vec, sen_e2_vec, des_vec_, vec_idx_arr = \
            #     self.build_classifaction_input(sen_e1, sen_e2, self.des_vectors_feature, training=False, top_k_indices=top_k_indices)

            # # print(des_vec_.shape)

            # sen_bs = sen_e1.size(0)
            # sen_local = sen_local_feature.view(sen_bs, 1, self.config.hidden_size).expand(-1, self.k, -1)
            x = torch.cat([sen_e1*sen_e2, sen_e1, sen_e2], dim=-1)
            hidden = F.relu(self.encoder_fc1(x))
            mu = self.encoder_fc_mu(hidden)
            logvar = self.encoder_fc_logvar(hidden)

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) # 从 N(0, I) 采样
            z = mu + eps * std
            # print(z.shape)
            y = torch.cat([z, sen_e1, sen_e2], dim=-1)
            hidden = F.relu(self.decoder_fc1(y))
            reconstructed_emb_concat = self.decoder_fc_out(hidden) # sigmoid 或 tanh 如果原始emb被归一化
            class_feature = torch.cat([reconstructed_emb_concat, sen_e1, sen_e2],dim=-1)
            des_class_feature = torch.mean(self.des_vectors)
            class_cos_sim = self.cos(class_feature.unsqueeze(1), self.des_vectors.unsqueeze(0))
            max_cls_sim, max_cls_sim_idx = torch.max(class_cos_sim, dim=1)
            # cls_feature = class_feature.reshape(batch_size, 4*self.k *self.config.hidden_size)
            # x1 = self.classify1(cls_feature)
            # x2 = self.classify2(torch.relu(x1))
            # logits = self.classify3(torch.relu(x2))

            # max_idx = torch.argmax(logits, dim=1)
            # # 由于max_idx（数量为k）和真实的idx（数量为unseen）不对应，所以需要转换
            # converted_max_idx = vec_idx_arr[torch.arange(vec_idx_arr.size(0)), max_idx]

            # input_ids, att_masks, vec_idx_arr, sen_vec, des_vec = \
            #     self.build_classifaction_input(sen_input_ids, self.des_input_ids_for_predict, sen_vec, self.des_vectors, training=False, top_k_indices=top_k_indices)
            # print(f'input_ids: {input_ids.shape}')
            # print(f'att_masks: {att_masks.shape}')
            # print(f'token_type_ids: {token_type_ids.shape}')
            # print(f'vec_idx_arr: {vec_idx_arr.shape}')
            # print(f'sen_vec: {sen_vec.shape}')
            # print(f'des_vec: {des_vec.shape}')
            # max_classify_idx, class_feature = self.classify(input_ids, att_masks, vec_idx_arr, sen_vec, des_vec)
            # print(f'max_sim_idx: {max_sim_idx}')
            # outputs = (outputs,) + max_sim_idx 
            return max_sim_idx, max_cls_sim_idx, sen_vec, self.des_vectors, reconstructed_emb_concat
            # return max_sim_idx, converted_max_idx, sen_vec, self.des_vectors
        # return outputs
    
    
    def gen_des_vectors(self, des_features):
        # print(des_features.shape)
        max_len = des_features.shape[1]
        des_input_ids = des_features[:, 0: int(max_len / 2)]
        self.des_input_ids_for_predict = des_input_ids
        des_attention_masks = des_features[:, int(max_len / 2): max_len]

        des_outputs = self.bert(
                input_ids=des_input_ids,
                attention_mask=des_attention_masks,
            )
        des_output = des_outputs.last_hidden_state
        self.des_vectors_feature = torch.mean(des_output,dim=1)
        self.des_vectors = self.get_des_vec(des_output)

    def get_sen_vec(self, sen_output, marked_e1, marked_e2):
        if self.add_auto_match:
            e1_h = extract_entity(sen_output, marked_e1) # [E1] [bs, hs]
            e2_h = extract_entity(sen_output, marked_e2) # [E2]
            sen_cls = torch.mean(sen_output,dim=1) # [bs, hs]
            # e1_h = torch.unsqueeze(e1_h, dim=1)
            # e2_h = torch.unsqueeze(e2_h, dim=1)
            # sen_cls = torch.unsqueeze(e1_h, dim=1) # [bs, 1, hs]
            # sen_vec = self.sen_att(e1_h, e2_h, sen_cls)
            # sen_vec = torch.squeeze(sen_vec, dim=1) # [bs, hs]
            sen_vec = torch.cat([sen_cls, e1_h, e2_h], dim=1) # [bs, 3* hs]
        else:
            sen_vec = torch.mean(sen_output,dim=1) # [cls]
        return sen_vec, e1_h, e2_h
    
    def get_sen_entity_between_vec(self, sen_output, marked_e1, marked_e2):

        e1_idx = torch.nonzero(marked_e1==1)
        e1_idx = torch.tensor([e1_idx[i][1] for i in range(len(e1_idx))])
        
        e2_idx = torch.nonzero(marked_e2==1)
        e2_idx = torch.tensor([e2_idx[i][1] for i in range(len(e2_idx))])     

        for i in range(len(e2_idx)):
            if e1_idx[i] < e2_idx[i]:
                marked_e1[i][e1_idx[i]:e2_idx[i]+1] = 1 
            else:
                marked_e1[i][e2_idx[i]:e1_idx[i]+1] = 1 

        extended_e_mask = marked_e1.unsqueeze(-1)
        # print(extended_e_mask.size()) # [32, 128, 1]
        extended_e_mask = extended_e_mask.float() * sen_output
        extended_e_mask = torch.mean(extended_e_mask,dim=-2)
        # print(extended_e_mask.size()) # [32, 768]
        # extended_e_mask = torch.stack([sequence_output[i,j,:] for i,j in enumerate(e_mask)])
        # print(extended_e_mask)
        return extended_e_mask.float()

    def get_des_vec(self, des_output):
        if self.add_auto_match:
            # [bs, ml, hs] x [hs, 1]
            des_cls = torch.mean(des_output,dim=1)
            des_output = des_output[:, 1:, :]
            bert_layer1 = torch.squeeze(torch.add(torch.matmul(des_output, self.des_weights1), self.des_bias1), dim=-1) #[bs, ml-1]
            bert_layer_softmax1 = torch.softmax(bert_layer1, dim=-1) # [bs, ml-1]
            e1_h = torch.sum(torch.unsqueeze(bert_layer_softmax1, dim=2) * des_output, dim=1) # [bs, ml-1, 1] * [bs, ml-1, hs]
            bert_layer2 = torch.squeeze(torch.add(torch.matmul(des_output, self.des_weights2), self.des_bias2), dim=-1) #[bs, ml-1]
            bert_layer_softmax2 = torch.softmax(bert_layer2, dim=-1) # [bs, ml - 1]
            e2_h = torch.sum(torch.unsqueeze(bert_layer_softmax2, dim=2) * des_output, dim=1)
            des_vec = torch.cat([des_cls, e1_h, e2_h], dim=1) # [bs, 3* hs]
        else:
            des_vec = torch.mean(des_output,dim=1)
        return des_vec

    def build_classifaction_input(self, sen_e1, sen_e2, des_vec_feature, training=True, top_k_indices=None, reconstructed_emb_concat=None):
        '''
        sen_input_ids: [bs, max_seq_length]
        des_input_ids: [bs, max_seq_length](train) or [k, max_seq_length](predict)
        sen_vec: [bs, hs]
        des_vec: [bs, hs](train) or [k, hs](predict)
        training: for train or predict, True for train, False for predict
        '''
        # print(f'build_classifaction_input :: training: {training}')
        device = sen_e1.device
        k = self.k
        batch_size = sen_e1.size(0)
        # 选取负样本的策略：随机k个 or vec最接近的k个


        sen_e1s = [] # [bs, k, ml]
        sen_e2s = [] # [bs, k, ml]
        des_vec_features = [] # [bs, k, ml]
        reconstructed_emb_concats = [] # [bs, k, ml]
        target_idx_arr = [] # [bs]
        vec_idx_arr = [] # [bs, k]
        # print(f'batch_size:{batch_size}')
        for idx in range(batch_size):
            sen_e1_per_sample = [] # [k, ml]
            sen_e2_per_sample = [] # [k, ml]
            des_vec_feature_per_sample = [] # [k, ml]
            reconstructed_emb_concat_per_sample = [] # [k, ml]
            vec_idx_arr_per_sample = []

            if training:
                # 随机添加，保证分类模型类别均衡
                target_idx = random.randint(0, k - 1)
            
            current_idx = 0
            while(current_idx < k):

                if training:
                    if current_idx == target_idx:
                        sen_e1_per_sample.append(sen_e1[idx])
                        sen_e2_per_sample.append(sen_e2[idx])
                        des_vec_feature_per_sample.append(des_vec_feature[idx])
                        reconstructed_emb_concat_per_sample.append(reconstructed_emb_concat[idx])
                        vec_idx_arr_per_sample.append(idx)
                    else:
                        # 随机选取负样本
                        neg_idx = random.randint(0, batch_size - 1)
                        # 跳过正样本本身
                        if neg_idx == idx:
                            continue
                        # else:
                        #     neg_idx = 
                        sen_e1_per_sample.append(sen_e1[idx])
                        sen_e2_per_sample.append(sen_e2[idx])
                        des_vec_feature_per_sample.append(des_vec_feature[neg_idx])
                        reconstructed_emb_concat_per_sample.append(reconstructed_emb_concat[idx])
                else:
                    vec_idx_arr_per_sample.append(top_k_indices[idx][current_idx])
                    sen_e1_per_sample.append(sen_e1[idx])
                    sen_e2_per_sample.append(sen_e2[idx])
                    des_vec_feature_per_sample.append(des_vec_feature[top_k_indices[idx][current_idx]])
                    # reconstructed_emb_concat_per_sample.append(reconstructed_emb_concat[idx])
                    
                current_idx += 1
            
            sen_e1s.append(torch.stack(sen_e1_per_sample,dim=0))
            sen_e2s.append(torch.stack(sen_e2_per_sample,dim=0))
            des_vec_features.append(torch.stack(des_vec_feature_per_sample,dim=0)) # [bs, k, ml]
            vec_idx_arr.append(vec_idx_arr_per_sample)
            
            if training:
                target_idx_arr.append(target_idx)
                reconstructed_emb_concats.append(torch.stack(reconstructed_emb_concat_per_sample,dim=0)) # [bs, k, ml]        
        sen_e1s = torch.stack(sen_e1s, dim=0).to(device)
        sen_e2s = torch.stack(sen_e2s, dim=0).to(device)
        des_vec_features = torch.stack(des_vec_features, dim=0).to(device)
        vec_idx_arr = torch.tensor(vec_idx_arr).to(device)
        if training:
            target_idx_arr = torch.tensor(target_idx_arr).to(device)
            reconstructed_emb_concats = torch.stack(reconstructed_emb_concats, dim=0).to(device)

        if training:
            return sen_e1s, sen_e2s, reconstructed_emb_concats, des_vec_features, target_idx_arr, vec_idx_arr
        else:
            return sen_e1s, sen_e2s, des_vec_features, vec_idx_arr

        '''
        sen_input_ids: [bs, max_seq_length]
        des_input_ids: [bs, max_seq_length](train) or [k, max_seq_length](predict)
        sen_vec: [bs, hs]
        des_vec: [bs, hs](train) or [k, hs](predict)
        training: for train or predict, True for train, False for predict
        '''
        # print(f'build_classifaction_input :: training: {training}')
        device = sen_e1.device
        k = self.k
        batch_size = sen_e1.size(0)
        # 选取负样本的策略：随机k个 or vec最接近的k个


        sen_e1s = [] # [bs, k, ml]
        sen_e2s = [] # [bs, k, ml]
        des_vec_features = [] # [bs, k, ml]
        reconstructed_emb_concats = [] # [bs, k, ml]
        target_idx_arr = [] # [bs]
        vec_idx_arr = [] # [bs, k]
        # print(f'batch_size:{batch_size}')
        for idx in range(batch_size):
            sen_e1_per_sample = [] # [k, ml]
            sen_e2_per_sample = [] # [k, ml]
            des_vec_feature_per_sample = [] # [k, ml]
            reconstructed_emb_concat_per_sample = [] # [k, ml]
            vec_idx_arr_per_sample = []

            if training:
                # 随机添加，保证分类模型类别均衡
                target_idx = random.randint(0, k - 1)
            
            current_idx = 0
            while(current_idx < k):

                if training:
                    if current_idx == target_idx:
                        sen_e1_per_sample.append(sen_e1[idx])
                        sen_e2_per_sample.append(sen_e2[idx])
                        des_vec_feature_per_sample.append(des_vec_feature[idx])
                        reconstructed_emb_concat_per_sample.append(reconstructed_emb_concat[idx])
                        vec_idx_arr_per_sample.append(idx)
                    else:
                        # 随机选取负样本
                        neg_idx = random.randint(0, batch_size - 1)
                        # 跳过正样本本身
                        if neg_idx == idx:
                            continue
                        # else:
                        #     neg_idx = 
                        sen_e1_per_sample.append(sen_e1[idx])
                        sen_e2_per_sample.append(sen_e2[idx])
                        des_vec_feature_per_sample.append(des_vec_feature[neg_idx])
                        reconstructed_emb_concat_per_sample.append(reconstructed_emb_concat[idx])
                else:
                    vec_idx_arr_per_sample.append(top_k_indices[idx][current_idx])
                    sen_e1_per_sample.append(sen_e1[idx])
                    sen_e2_per_sample.append(sen_e2[idx])
                    des_vec_feature_per_sample.append(top_k_indices[idx][current_idx])
                    reconstructed_emb_concat_per_sample.append(reconstructed_emb_concat[idx])
                    
                current_idx += 1
            
            sen_e1s.append(torch.stack(sen_e1_per_sample,dim=0))
            sen_e2s.append(torch.stack(sen_e2_per_sample,dim=0))
            des_vec_features.append(torch.stack(des_vec_feature_per_sample,dim=0)) # [bs, k, ml]
            reconstructed_emb_concats.append(torch.stack(reconstructed_emb_concat_per_sample,dim=0)) # [bs, k, ml]
            vec_idx_arr.append(vec_idx_arr_per_sample)
            
            if training:
                target_idx_arr.append(target_idx)
        
        sen_e1s = torch.stack(sen_e1s, dim=0).to(device)
        sen_e2s = torch.stack(sen_e2s, dim=0).to(device)
        des_vec_features = torch.stack(des_vec_features, dim=0).to(device)
        reconstructed_emb_concats = torch.stack(reconstructed_emb_concats, dim=0).to(device)
        vec_idx_arr = torch.tensor(vec_idx_arr).to(device)
        if training:
            target_idx_arr = torch.tensor(target_idx_arr).to(device)

        if training:
            return sen_e1s, sen_e2s, des_vec_features, reconstructed_emb_concats, target_idx_arr, vec_idx_arr
        else:
            return sen_e1s, sen_e2s, des_vec_features, reconstructed_emb_concats, vec_idx_arr

class Classify_model(nn.Module): 
    def __init__(self, pretrain_model_name_or_path, max_seq_len=128, k=3):
        super(Classify_model, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrain_model_name_or_path)
        self.bert = AutoModel.from_pretrained(pretrain_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name_or_path)
        self.max_seq_len = max_seq_len
        self.k = k
        self.mlp1 = nn.Linear(k * self.config.hidden_size, self.config.hidden_size)
        self.mlp2 = nn.Linear(self.config.hidden_size, k)

        # self.mha = MultiHeadAttention(self.config.hidden_size, 0.1, 8)

    def forward(self, input_ids, att_masks, vec_idx_arr, sen_vec_arr, des_vec_arr, target_idx_arr=None):
        '''
        input_ids: [bs, k, ml]
        att_masks: [bs, k, ml]
        token_type_ids: [bs, k, ml]
        target_idx_arr: [bs]
        vec_idx_arr: [bs, k]
        sen_vec: [bs, hs]
        des_vec: [bs, hs]
        '''
        # print(vec_idx_arr)
        batch_size = len(input_ids)
        flatten_input_ids = input_ids.view(-1, self.max_seq_len) # [bs * k, ml]
        flatten_att_masks = att_masks.view(-1, self.max_seq_len) # [bs * k, ml]
        # flatten_token_type_ids = token_type_ids.view(-1, self.max_seq_len) # [bs * k, ml]
        outputs = self.bert(
                input_ids=flatten_input_ids,
                attention_mask=flatten_att_masks,
                # token_type_ids=flatten_token_type_ids
            )
        bert_output = outputs.last_hidden_state
        bert_output = bert_output[:, 0, :] # [bs * k, hs]
        # bert_output = torch.unsqueeze(bert_output, dim=1) # [bs * k, 1, hs]
        # bert_output = bert_output.view(batch_size, self.k, self.config.hidden_size) # [bs, k, hs]
        bert_output = bert_output.reshape(batch_size, self.k * self.config.hidden_size) # [bs, k * hs]
        # 根据vec_idx_arr取出对应的vec
        # flatten_idx_arr = vec_idx_arr.view(-1) # [bs * k]
        # sen_vec_s = torch.index_select(sen_vec_arr, 0, flatten_idx_arr) # [bs * k, hs]
        # des_vec_s = torch.index_select(des_vec_arr, 0, flatten_idx_arr) # [bs * k, hs]
        
        # # # sen_vec_s = torch.unsqueeze(sen_vec_s, dim=1) # [bs * k, 1, hs]
        # # # des_vec_s = torch.unsqueeze(des_vec_s, dim=1) # [bs * k, 1, hs]

        # sen_vec_s = sen_vec_s.view(batch_size, self.k, self.config.hidden_size) # [bs, k, hs]
        # des_vec_s = des_vec_s.view(batch_size, self.k, self.config.hidden_size) # [bs, k, hs]

        # # BERT output 和 DSSM vec交互方式
        # sen_att_output = self.mha(sen_vec_s, bert_output, bert_output) # [bs, k, hs]
        # des_att_output = self.mha(des_vec_s, bert_output, bert_output) # [bs, k, hs]

        # sen_att_output = sen_att_output.view(batch_size, self.k * self.config.hidden_size) # [bs, k * hs]
        # des_att_output = des_att_output.view(batch_size, self.k * self.config.hidden_size) # [bs, k * hs]

        # concatenated_output = torch.cat([sen_att_output, des_att_output], dim=1) # [bs, 2k * hs]

        
        x_feature = self.mlp1(bert_output) # [bs, hs]
        # x = self.mlp1(concatenated_output) # [bs, hs]
        x = torch.relu(x_feature) # [bs, hs]
        logits = self.mlp2(x) # [bs, k]
        # print(f'logits: {logits}')
        # print(f'target_idx_arr: {target_idx_arr}')
        if target_idx_arr != None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, target_idx_arr)
            # print(f'target_idx_arr: {target_idx_arr}')
            return loss
        else:
            max_idx = torch.argmax(logits, dim=1)
            # 由于max_idx（数量为k）和真实的idx（数量为unseen）不对应，所以需要转换
            converted_max_idx = vec_idx_arr[torch.arange(vec_idx_arr.size(0)), max_idx]
            # print(f'vec_idx_arr:{vec_idx_arr}')
            # print(f'max_idx: {max_idx}')
            # print(f'converted_max_idx: {converted_max_idx}')
            return converted_max_idx, x_feature

# class MultiHeadAttention(nn.Module):

#     def __init__(self, hidden_size, dropout_rate, head_size=8) -> None:
#         super(MultiHeadAttention, self).__init__()
        
#         self.head_size = head_size
#         self.att_size = att_size = hidden_size // head_size
        
#         self.scale = att_size ** -0.5

#         self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
#         self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
#         self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)

#         self.att_dropout = nn.Dropout(dropout_rate)

#         self.output_layer = nn.Linear(head_size * att_size, hidden_size, bias=False)

#     def forward(self, q, k, v, cache=None):
#         orig_q_size = q.size()

#         d_k = self.att_size
#         d_v = self.att_size
#         batch_size = q.size(0)

#         q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
#         k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
#         v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

#         q = q.transpose(1,2)
#         v = v.transpose(1,2)
#         k = k.transpose(1,2).transpose(2,3)

#         q.mul_(self.scale)
#         x = torch.matmul(q, k)
#         # x.masked_fill_(mask.unsqueeze(1), -1e9)

#         x = torch.softmax(x, dim=-1)
#         x = self.att_dropout(x)
#         x = x.matmul(v)

#         x = x.transpose(1,2).contiguous()
#         x = x.view(batch_size, -1, self.head_size*d_v)

#         x = self.output_layer(x)

#         assert x.size() == orig_q_size
#         return x
