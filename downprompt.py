import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL
from layers import Attentivemod

from layers import GCN, AvgReadout
import torch_scatter
from tools import *

class prefeatureprompt(nn.Module):
    def __init__(self,texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,sumtext,dim,type:str,head_num=8):
        super(prefeatureprompt, self).__init__()
        self.precomposedfeature = composedtoken(texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,type)
        self.preopenfeature = downstreamprompt(dim)
        self.sumtext = sumtext
        self.combineprompt = combineprompt()
    def forward(self,seq):
        seq1 = self.precomposedfeature(seq)
        seq1 = F.relu(seq1)
        seq3 = self.sumtext * seq1
        seq2 = self.preopenfeature(seq)
        ret =  self.combineprompt(seq3 ,seq2) 
        return ret

class composedtoken(nn.Module):
    def __init__(self,texttoken1,texttoken2,texttoken3,texttoken4,texttoken5,type:str,head_num=8):
        super(composedtoken, self).__init__()   
        self.texttoken = torch.cat((texttoken1,texttoken2,texttoken3,texttoken4,texttoken5),dim=0)
        self.prompt = weighted_prompt(5).cuda()
        self.type = type

    def forward(self,seq):
        texttoken = self.prompt(self.texttoken)
        if self.type == 'add':
            texttoken = texttoken.repeat(seq.shape[0],1)
            rets = texttoken + seq
        if self.type == 'mul':
            rets = texttoken * seq
        return rets

class ATT_learner(nn.Module):
    def __init__(self, nlayers, isize, i, dropedge_rate, sparse, act):
        super(ATT_learner, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Attentivemod.Attentive(isize))

        self.non_linearity = 'relu'
        self.i = i
        self.sparse = sparse
        self.act = act
        self.dropedge_rate = dropedge_rate

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)

        return h

    def forward(self, features):
       
        embeddings = self.internal_forward(features)

        return embeddings

    def graph_process(self, k, embeddings):
        if self.sparse:
            rows, cols, values = knn_fast(embeddings, k, 100)      
            values[torch.isnan(values)] = 0  
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            values_ = F.dropout(values_, p=self.dropedge_rate, training=self.training)
            num_nodes = embeddings.shape[0]
            learned_adj = torch.zeros((num_nodes, num_nodes), device='cuda') 
            learned_adj[rows_, cols_] = values_ 

            return learned_adj
        else:
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, k + 1)
            similarities = symmetrize(similarities)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            learned_adj = normalize(similarities, 'sym')
            learned_adj = F.dropout(learned_adj, p=self.dropedge_rate, training=self.training)

            return learned_adj

class textprompt(nn.Module):
    def __init__(self,hid_units,type):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.prompttype =type
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding):
        if self.prompttype == 'add':
            weight = self.weight.repeat(graph_embedding.shape[0],1)
            graph_embedding = weight + graph_embedding
        if self.prompttype == 'mul':
            graph_embedding = self.weight * graph_embedding

        return graph_embedding

class downprompt(nn.Module):
    def __init__(self,token1,token2,token3,token4,token5,
            sumtext,pretoken1,pretoken2,pretoken3,pretoken4,pretoken5,balancetoken1,balancetoken2,balancetoken3,balancetoken4,balancetoken5,
            ft_in, nb_classes, type,feature_dim):
        super(downprompt, self).__init__()
        self.downprompt = downstreamprompt(ft_in)
        self.composedprompt = composedtoken(token1,token2,token3,token4,token5,type=type)
        self.prefeature = prefeatureprompt(pretoken1,pretoken2,pretoken3,pretoken4,pretoken5,sumtext,dim=feature_dim,type=type,head_num=4)
        self.combineprompt1 = combineprompt()
        self.combineprompt2 = combineprompt()
        self.balancedprompt = textprompt(2 * feature_dim,type)
        self.learner = ATT_learner(2, 50, 6, 0.5, sparse = True, act = 'relu')

        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU() 
        self.one = torch.ones(1,ft_in).cuda()
        self.ave = torch.FloatTensor(nb_classes,ft_in).cuda()


    def forward(self,features,adj,sparse,gcn,idx,seq,downk,labels=None,train=0):

        features1 = self.prefeature(features)
        reseq1 = torch.sparse.mm(adj,features1)
        reseq1 = torch.cat((features1, reseq1), dim = 1)
        reseq111 = self.balancedprompt(reseq1) 
        adj1 = self.learner.graph_process(downk, reseq111).to('cuda')
        alpha = torch.nn.Parameter(torch.tensor(0.5))
        adjtot = alpha * adj.to_dense() + (1 - alpha) * adj1
        embeds1 = gcn(features1, adjtot, sparse, None).squeeze()
        pretrain_embs1 = embeds1[idx]        
        rawret = pretrain_embs1
        rawret = rawret.cuda()
        
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret,nb_class=self.nb_classes)
        
        ret = torch.FloatTensor(seq.shape[0],self.nb_classes).cuda()
        
        rawret = torch.cat((rawret,self.ave),dim=0)
        
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:seq.shape[0],seq.shape[0]:]
                
        ret = F.softmax(ret, dim=1)

        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

class combineprompt(nn.Module):
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)


    def forward(self, graph_embedding1, graph_embedding2):
        
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)

def averageemb(labels,rawret,nb_class):
    retlabel = torch_scatter.scatter(src=rawret,index=labels,dim=0,reduce='mean')
    return retlabel


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        x = self.fc(attended_values) + x
        x = torch.squeeze(x)

        x = torch.sum(x,dim=0)
        return x


class weighted_prompt(nn.Module):
    def __init__(self,weightednum):
        super(weighted_prompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding):

        graph_embedding=torch.mm(self.weight,graph_embedding)
        return graph_embedding
   

class weighted_feature(nn.Module):
    def __init__(self,weightednum):
        super(weighted_feature, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)
    def forward(self, graph_embedding1,graph_embedding2):
        graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)
    
class downstreamprompt(nn.Module):
    def __init__(self,hid_units):
        super(downstreamprompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding):
        graph_embedding=self.weight * graph_embedding
        return graph_embedding
    
class featureprompt(nn.Module):
    def __init__(self,prompt1,prompt2,prompt3):
        super(featureprompt, self).__init__()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        self.weightprompt = weighted_prompt(3)
    def forward(self,feature):
        weight = self.weightprompt(self.prompt)
        feature = weight * feature
        return feature