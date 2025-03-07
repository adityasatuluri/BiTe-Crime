import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# Built on original code base of (Check layers_torch.py): 
# https://github.com/nke001/sparse_attentive_backtracking_release
class Sparse_attention(nn.Module):
    def __init__(self, top_k=5):
        super(Sparse_attention, self).__init__()
        self.top_k = top_k

    def forward(self, attn_s):
        eps = 10e-8
        batch_size = attn_s.size()[0]
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            return attn_s
        else:
            delta = torch.topk(attn_s, self.top_k, dim=1)[0][:, -1] + eps

        # Dynamic sparsity: Retain most significant weights
        attn_w = torch.clamp(attn_s - delta.view(batch_size, 1).repeat(1, time_step), min=0)
        attn_w_sum = attn_w.sum(dim=1, keepdim=True) + eps
        return attn_w / attn_w_sum


# Built on original code base of (Check layers_torch.py): 
# https://github.com/nke001/sparse_attentive_backtracking_release
class self_LSTM_sparse_attn_predict(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 truncate_length=100, predict_m=10, block_attn_grad_past=False, attn_every_k=1, top_k=5):
        super(self_LSTM_sparse_attn_predict, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.attn_every_k = attn_every_k
        self.top_k = top_k
        self.tanh = torch.nn.Tanh()

        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        nn.init.xavier_uniform_(self.w_t.data, gain=1.414)

        self.sparse_attn = Sparse_attention(top_k=self.top_k)
        self.predict_m = nn.Linear(hidden_size, 2)  # hidden_size

    def forward(self, x):
        batch_size = x.size(0)
        time_size = x.size(1)
        input_size = self.input_size
        hidden_size = self.hidden_size

        h_t = Variable(torch.zeros(batch_size, hidden_size))  # h_t = (batch_size, hidden_size)
        c_t = Variable(torch.zeros(batch_size, hidden_size))  # c_t = (batch_size, hidden_size)
        predict_h = Variable(torch.zeros(batch_size, hidden_size))  # predict_h = (batch_size, hidden_size)

        h_old = h_t.view(batch_size, 1, hidden_size)  # h_old = (batch_size, 1, hidden_size) --> Memory

        outputs = []
        attn_all = []
        attn_w_viz = []
        predicted_all = []
        outputs_new = []

        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            remember_size = h_old.size(1)

            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = h_t.detach(), c_t.detach()

            # Feed LSTM Cell
            input_t = input_t.contiguous().view(batch_size, input_size)  # input_t = (batch_size, input_size)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))  # h_t/ c_t = (batch_size, hidden dimension)
            h_t_naive_lstm = h_t
            predict_h = self.predict_m(h_t.detach())  # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
            predicted_all.append(h_t)  # changed predict_h

            # Broadcast and concatenate current hidden state against old states
            h_repeated = h_t.unsqueeze(1).repeat(1, remember_size,
                                                 1)  # h_repeated = (batch_size, remember_size = memory, hidden_size)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)  # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)

            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            mlp_h_attn = self.tanh(mlp_h_attn)  # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)

            if False:  # PyTorch 0.2.0
                attn_w = torch.matmul(mlp_h_attn, self.w_t)
            else:  # PyTorch 0.1.12
                mlp_h_attn = mlp_h_attn.view(batch_size * remember_size,
                                             2 * hidden_size)  # mlp_h_attn = (batch_size * remember_size, 2* hidden_size)
                attn_w = torch.mm(mlp_h_attn, self.w_t)  # attn_w = (batch_size * remember_size, 1)
                attn_w = attn_w.view(batch_size, remember_size, 1)  # attn_w = (batch_size, remember_size, 1)
            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w = attn_w.view(batch_size, remember_size)  # attn_w = (batch_size, remember_size)
            attn_w = self.sparse_attn(attn_w)  # attn_w = (batch_size, remember_size)
            attn_w = attn_w.view(batch_size, remember_size, 1)  # attn_w = (batch_size, remember_size, 1)

            # if i >= 100:
            # print(attn_w.mean(dim=0).view(remember_size))
            attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))  # you should return it
            out_attn_w = attn_w
            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w = attn_w.repeat(1, 1, hidden_size)  # attn_w = (batch_size, remember_size, hidden_size)
            h_old_w = attn_w * h_old  # attn_w = (batch_size, remember_size, hidden_size)
            attn_c = torch.sum(h_old_w, 1).squeeze(1)  # att_c = (batch_size, hidden_size)

            # Feed attn_c to hidden state h_t
            h_t = h_t + attn_c  # h_t = (batch_size, hidden_size)

            #
            # At regular intervals, remember a hidden state, store it in memory
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)

            predict_real_h_t = self.predict_m(
                h_t.detach())  # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
            outputs_new += [predict_real_h_t]

            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        return attn_c, out_attn_w


# Built on original code base of (Check layers.py): 
# https://github.com/Diego999/pyGAT
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_cfeatures, in_xfeatures, out_features, att_dim, bs, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_cfeatures = in_cfeatures
        self.in_xfeatures = in_xfeatures
        self.out_features = out_features
        self.att_dim = att_dim
        self.bs = bs
        self.emb_dim = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_cfeatures, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.Wf = nn.Parameter(torch.zeros(size=(in_cfeatures, out_features)))
        nn.init.xavier_uniform_(self.Wf.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.WS = nn.Parameter(torch.zeros(size=(in_cfeatures, out_features)))
        nn.init.xavier_uniform_(self.WS.data, gain=1.414)

        self.aS = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.aS.data, gain=1.414)

        self.WQ = nn.Parameter(torch.zeros(size=(2, self.att_dim)))
        nn.init.xavier_uniform_(self.WQ.data, gain=1.414)

        self.WK = nn.Parameter(torch.zeros(size=(2, self.att_dim)))
        nn.init.xavier_uniform_(self.WK.data, gain=1.414)

        self.WV = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.WV.data, gain=1.414)

        self.WF = nn.Linear(self.in_xfeatures, out_features, bias=False)  # For other one

    def forward(self, input, adj, ext_input, side_input):
        input = input.view(self.bs, -1, 1)
        ext_input = ext_input.view(self.bs, -1, self.in_xfeatures)
        side_input = side_input.view(self.bs, -1, 1)
        adj = adj.repeat(self.bs, 1, 1)

        """
        Step 1: Generate c_{i,t}^k using GATv2
        """
        # GATv2 implementation
        h = torch.matmul(input, self.W)  # h = [h_1, h_2, h_3, ... , h_N] * W
        N = h.size()[1]  # N = Number of Nodes (regions)
        a_input = torch.cat([h.repeat(1, 1, N).view(h.shape[0], N * N, -1), h.repeat(1, N, 1)], dim=2).view(h.shape[0], N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # The key difference in GATv2: apply leaky ReLU before multiplication with attention weights
        e = self.leakyrelu(e)
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Find the attention vectors for side_wise crime similarity (GATv2 style)
        h_side = torch.matmul(side_input, self.WS)
        a_input_side = torch.cat([h_side.repeat(1, 1, N).view(self.bs, N * N, -1), h_side.repeat(1, N, 1)], dim=2).view(self.bs, N, -1, 2 * self.out_features)
        e_side = self.leakyrelu(torch.matmul(a_input_side, self.aS).squeeze(3))
        
        # Apply leaky ReLU before multiplication with attention weights
        e_side = self.leakyrelu(e_side)
        
        attention_side = torch.where(adj > 0, e_side, zero_vec)
        attention_side = F.dropout(attention_side, self.dropout, training=self.training)

        # Combine attentions
        attention = attention + attention_side
        attention = torch.where(attention > 0, attention, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        """
            Step 2: Generate e_{i,t}^k
        """
        # Generate Query Vector
        q = torch.cat([input.repeat(1, 1, N).view(input.shape[0], N * N, -1), input.repeat(1, N, 1)], dim=2).view(
            input.shape[0], N, N, -1)
        q = torch.matmul(q, self.WQ)  # (bs, N, N, dq) = (bs, N, N, 2) * (2, dq)
        q = q / (self.att_dim ** 0.5)
        q = q.unsqueeze(3)  # (bs, N, N, 1, dq)

        # Generate Key Vector
        ext_input = ext_input.unsqueeze(3)
        k = torch.cat([ext_input.repeat(1, 1, N, 1).view(ext_input.shape[0], N * N, self.in_xfeatures, -1),
                       ext_input.repeat(1, N, 1, 1).view(ext_input.shape[0], N * N, self.in_xfeatures, -1)], dim=3). \
            view(ext_input.shape[0], N, N, self.in_xfeatures, 2)
        k = torch.matmul(k, self.WK)  # (bs, N, N, in_xfeatures, dk) = (bs, N, N, in_xfeatures, 2)* (2, dk)
        k = torch.transpose(k, 4, 3)  # (bs, N, N, dk, in_xfeatures)

        # Generate Value Vector
        v = torch.matmul(ext_input, self.WV)  # (bs, N, N, in_xfeatures, dv)

        # Generate dot product attention
        dot_attention = torch.matmul(q, k).squeeze(3)  # (bs, N, N, in_xfeatures)
        zero_vec = -9e15 * torch.ones_like(dot_attention)
        dot_attention = torch.where(dot_attention > 0, dot_attention, zero_vec)  # (bs, N, N, in_xfeatures)
        dot_attention = F.softmax(dot_attention, dim=3)  # shape = (bs, N, N, in_xfeatures)

        # Generate the external feature representation of the regions: e_{i,t}^k
        crime_attention = attention.unsqueeze(3).repeat(1, 1, 1, self.in_xfeatures)
        final_attention = dot_attention * crime_attention
        ext_rep = torch.matmul(final_attention, v)  # (bs, N, N, dv)
        ext_rep = ext_rep.sum(dim=2)  # (bs, N, N, dv)

        if self.concat:
            return F.elu(h_prime), F.elu(ext_rep)
        else:
            return h_prime, ext_rep

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_cfeatures) + ' -> ' + str(self.out_features) + ')'


#----------------------------------------------------------------------------------


class self_BiLSTM_sparse_attn_predict(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 truncate_length=100, predict_m=10, block_attn_grad_past=False, attn_every_k=1, top_k=5):
        super(self_BiLSTM_sparse_attn_predict, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length = truncate_length
        self.lstm_forward = nn.LSTMCell(input_size, hidden_size)
        self.lstm_backward = nn.LSTMCell(input_size, hidden_size)
        self.attn_every_k = attn_every_k
        self.top_k = top_k
        self.tanh = torch.nn.Tanh()

        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 4, 1))
        nn.init.xavier_uniform_(self.w_t.data, gain=1.414)

        self.sparse_attn = Sparse_attention(top_k=self.top_k)
        self.predict_m = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        batch_size, seq_length = x.shape
        input_size = x.size(1)  # Assuming each time step is a single value

        h_t_f = Variable(torch.zeros(batch_size, self.hidden_size))
        c_t_f = Variable(torch.zeros(batch_size, self.hidden_size))
        h_t_b = Variable(torch.zeros(batch_size, self.hidden_size))
        c_t_b = Variable(torch.zeros(batch_size, self.hidden_size))

        h_old = torch.cat((h_t_f, h_t_b), dim=1).view(batch_size, 1, self.hidden_size * 2)

        outputs = []
        attn_all = []
        attn_w_viz = []
        predicted_all = []
        outputs_new = []

        for i in range(seq_length):
            if (i + 1) % self.truncate_length == 0:
                h_t_f, c_t_f = h_t_f.detach(), c_t_f.detach()
                h_t_b, c_t_b = h_t_b.detach(), c_t_b.detach()

            input_forward = x[:, i].unsqueeze(1)
            input_backward = x[:, seq_length - 1 - i].unsqueeze(1)

            h_t_f, c_t_f = self.lstm_forward(input_forward, (h_t_f, c_t_f))
            h_t_b, c_t_b = self.lstm_backward(input_backward, (h_t_b, c_t_b))

            h_t_combined = torch.cat((h_t_f, h_t_b), dim=1)
            predicted_all.append(h_t_combined)

            remember_size = h_old.size(1)
            h_repeated = h_t_combined.unsqueeze(1).repeat(1, remember_size, 1)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)

            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            mlp_h_attn = self.tanh(mlp_h_attn)

            mlp_h_attn = mlp_h_attn.view(batch_size * remember_size, 4 * self.hidden_size)
            attn_w = torch.mm(mlp_h_attn, self.w_t)
            attn_w = attn_w.view(batch_size, remember_size, 1)

            attn_w = attn_w.view(batch_size, remember_size)
            attn_w = self.sparse_attn(attn_w)
            attn_w = attn_w.view(batch_size, remember_size, 1)

            attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))
            out_attn_w = attn_w

            attn_w = attn_w.repeat(1, 1, self.hidden_size * 2)
            h_old_w = attn_w * h_old
            attn_c = torch.sum(h_old_w, 1).squeeze(1)

            h_t_combined = h_t_combined + attn_c

            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t_combined.view(batch_size, 1, self.hidden_size * 2)), dim=1)

            predict_real_h_t = self.predict_m(h_t_combined.detach())
            outputs_new += [predict_real_h_t]

            outputs += [h_t_combined]
            attn_all += [attn_c]

        return attn_c, out_attn_w