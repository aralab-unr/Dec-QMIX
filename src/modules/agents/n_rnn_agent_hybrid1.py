import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class NRNNAgent(nn.Module):
    def __init__(self, n_vector_states, n_image_channel, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        self.bn_vec = nn.BatchNorm1d(1)
        self.dense1 = nn.Linear(n_vector_states, 50)
        self.dense2 = nn.Linear(50, 50)
        self.dense3 = nn.Linear(50, 50)

        self.bn_img = nn.BatchNorm2d(n_image_channel)
        self.conv1 = nn.Conv2d(n_image_channel, 4, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # We assume the flattened image features + 50-d vector gives 3250,
        # then we project to 256. (Same as CRADLE's main_dense_layer.)
        self.embed_linear = nn.Linear(3250, 256)

        # -------------------------------------------------
        # 2) RNN-based agent head (like NRNNAgent):
        #    (embedding -> fc1 -> GRUCell -> fc2 -> Q)
        # -------------------------------------------------
        self.fc1 = nn.Linear(256, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.embed_linear.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, vector_in, image_in, hidden_state):
        """
        vector_in:  [batch, n_agents, n_vector_states]
        image_in:   [batch, n_agents, channels, h, w]
        hidden_state: [batch, n_agents, rnn_hidden_dim] or [1, rnn_hidden_dim]
        
        Returns: (Q-values, next_hidden)
          Q-values shape:        [batch, n_agents, n_actions]
          next_hidden shape:     [batch, n_agents, rnn_hidden_dim]
        """

        b, a, vec_dim = vector_in.shape
        _, _, c, h, w = image_in.shape

        # Flatten batch & agent dims so [b*a, ...]
        vector_in = vector_in.view(-1, vec_dim)           # [b*a, vec_dim]
        image_in  = image_in.view(-1, c, h, w)            # [b*a, c, h, w]

        # -------------------------------------------------
        # 1) CRADLE-like front-end to get a 256-D embedding
        # -------------------------------------------------
        # Vector sub-network
        vec = vector_in.unsqueeze(1)                      # [b*a, 1, vec_dim]
        vec = self.bn_vec(vec).squeeze(1)                 # batch norm
        vec = F.relu(self.dense1(vec))
        vec = F.relu(self.dense2(vec))
        vec = F.relu(self.dense3(vec))

        # Image sub-network
        img = self.bn_img(image_in)
        img = self.maxpool(F.relu(self.conv1(img)))
        img = self.maxpool(F.relu(self.conv2(img)))
        img = self.maxpool(F.relu(self.conv3(img)))
        img = self.maxpool(F.relu(self.conv4(img)))
        img = img.view(img.size(0), -1)

        # Combine
        cradle_input = th.cat((vec, img), dim=1)  # shape [b*a, 50 + 3200 = 3250]
        embed = F.relu(self.embed_linear(cradle_input))   # [b*a, 256]

        # -------------------------------------------------
        # 2) RNN forward pass (like your NRNNAgent)
        # -------------------------------------------------
        x = F.relu(self.fc1(embed), inplace=True)
        h_in = hidden_state.view(-1, self.args.rnn_hidden_dim)  # [b*a, rnn_hid]
        h_out = self.rnn(x, h_in)                                # [b*a, rnn_hid]

        # optional layer norm
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(h_out))
        else:
            q = self.fc2(h_out)  # [b*a, n_actions]

        # Reshape back to [b, a, ...]
        q = q.view(b, a, -1)
        h_out = h_out.view(b, a, -1)
        return q, h_out