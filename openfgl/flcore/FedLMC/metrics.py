import torch
import torch.nn as nn
import torch.nn.functional as F

class EpsilonNN(nn.Module):
    '''
    Select elements according to the threshold.

    Parameters
    ----------
    epsilon : float
        Threshold.
    set_value : float
        Specify the value for selected elements. The original value will be used if set to `None`.
    '''

    def __init__(self, epsilon, set_value=None):
        super(EpsilonNN, self).__init__()
        self.epsilon = epsilon
        self.set_value = set_value

    def forward(self, adj):
        '''
        Generate matrix given adj matrix.
        Parameters
        ----------
        adj : torch.tensor
            Input adj. Note that either sparse or dense form is supported.

        Returns
        -------
        new_adj : torch.tensor
            Adj with elements larger than the threshold.
        '''
        return enn(adj, self.epsilon, self.set_value)


def enn(adj, epsilon, set_value=None):
    if adj.is_sparse:
        n = adj.shape[0]
        values = adj.values()
        mask = values > epsilon
        mask.requires_grad = False
        new_values = values[mask]
        if set_value:
            new_values[:] = set_value
        new_indices = adj.indices()[:, mask]
        return torch.sparse_coo_tensor(new_indices, new_values, [n, n])
    else:
        mask = adj > epsilon
        mask.requires_grad = False
        new_adj = adj * mask
        if set_value:
            new_adj[mask] = set_value
        return new_adj



def knn(adj, K, self_loop=True, set_value=None, sparse_out=False):
    if adj.is_sparse:
        # TODO
        pass
    else:
        device = adj.device
        values, indices = adj.topk(k=int(K), dim=-1)
        assert torch.max(indices) < adj.shape[1]
        if sparse_out:
            n = adj.shape[0]
            new_indices = torch.stack([torch.arange(n).view(-1, 1).expand(-1, int(K)).contiguous().flatten().to(device),
                                       indices.flatten()])
            new_values = values.flatten()
            return torch.sparse_coo_tensor(new_indices, new_values, [n, n]).coalesce()
        else:
            mask = torch.zeros(adj.shape).to(device)
            mask[torch.arange(adj.shape[0]).view(-1, 1), indices] = 1.
            if not self_loop:
                mask[torch.arange(adj.shape[0]).view(-1, 1), torch.arange(adj.shape[0]).view(-1, 1)] = 0
            mask.requires_grad = False
            new_adj = adj * mask
            if set_value:
                new_adj[new_adj.nonzero()[:, 0], new_adj.nonzero()[:, 1]] = set_value
            return new_adj

class WeightedCosine(nn.Module):
    '''
    Weighted cosine to generate pairwise similarities from given node embeddings.

    Parameters
    ----------
    d_in : int
        Dimensions of input features.
    num_pers : int
        Number of multi heads.
    weighted : bool
        Whether to use weighted cosine. cosine will be used if set to `None`.
    normalize : bool
        Whetehr to use normalize before multiplication.
    '''

    def __init__(self, d_in, num_pers=16, weighted=True, normalize=True):
        super(WeightedCosine, self).__init__()
        self.normalize = normalize
        self.w = None
        if weighted:
            self.w = nn.Parameter(torch.FloatTensor(num_pers, d_in))  # num_pers*feat_dim
            self.reset_parameters()
        #图结构只有1个，n*n，学16个图结构，从16个维度刻画
    def reset_parameters(self):
        if self.w is not None:
            nn.init.xavier_uniform_(self.w)

    def forward(self, x, y=None, non_negative=False):
        '''

        x:n*feat_shape
        x=linear(x)
        cos(x,x)-->n*n

        Given two groups of node embeddings, calculate the pairwise similarities.

        Parameters
        ----------
        x : torch.tensor
            Input features.
        y : torch.tensor
            Input features. ``x`` will be used if set to `None`.
        non_negative : bool
            Whether to mask negative elements.

        Returns
        -------
        adj : torch.tensor
            Pairwise similarities.
        '''

        if y is None:
            y = x
        context_x = x.unsqueeze(0)  # 1*num_node*feat_dim
        context_y = y.unsqueeze(0)
        if self.w is not None:
            expand_weight_tensor = self.w.unsqueeze(1)  # num_pers*1*feat_dim
            context_x = context_x * expand_weight_tensor  # (1*num_node*feat_dim)*(num_pers*1*feat_dim)，沿着1的维度广播，都扩展为 num_pers*num_node*feat_dim
            context_y = context_y * expand_weight_tensor
        if self.normalize:
            context_x = F.normalize(context_x, p=2, dim=-1)
            context_y = F.normalize(context_y, p=2, dim=-1)

        adj = torch.matmul(context_x, context_y.transpose(-1, -2)).mean(0)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)
        return adj

class Cosine(nn.Module):
    '''
    Cosine to generate pairwise similarities from given node embeddings.
    '''

    def __init__(self):
        super(Cosine, self).__init__()
        pass

    def forward(self, x, y=None, non_negative=False):
        '''
        Given two groups of node embeddings, calculate the pairwise similarities.

        Parameters
        ----------
        x : torch.tensor
            Input features.
        y : torch.tensor
            Input features. ``x`` will be used if set to `None`.
        non_negative : bool
            Whether to mask negative elements.

        Returns
        -------
        adj : torch.tensor
            Pairwise similarities.
        '''
        if y is None:
            y = x
        context_x = F.normalize(x, p=2, dim=-1)
        context_y = F.normalize(y, p=2, dim=-1)
        adj = torch.matmul(context_x, context_y.T)
        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)
        return adj

class KNN(nn.Module):
    '''
    Select KNN matrix each row.

    Parameters
    ----------
    K : int
        Number of neighbors.
    self_loop : bool
        Whether to include self loops.
    set_value : float
        Specify the value for selected elements. The original value will be used if set to `None`.
    metric : str
        The similarity function.
    sparse_out : bool
        Whether to return adj in sparse form.
    '''

    def __init__(self, K, self_loop=True, set_value=None, metric='cosine', sparse_out=False):
        super(KNN, self).__init__()
        self.K = K
        self.self_loop = self_loop
        self.set_value = set_value
        self.sparse_out = sparse_out
        if metric:
            if metric == 'cosine':
                self.metric = Cosine()

    def forward(self, x=None, adj=None):
        '''
        Generate KNN matrix given node embeddings or adj matrix. Pairwise similarities will first
        be calculated if ``x`` is given.
        Parameters
        ----------
        x : torch.tensor
            Node embeddings.
        adj : torch.tensor
            Input adj. Note only dense form is supported currently.

        Returns
        -------
        knn_adj : torch.tensor
            KNN matrix.
        '''
        assert not (x is None and adj is None)
        if x is not None:
            dist = self.metric(x)
        else:
            dist = adj
        return knn(dist, self.K, self.self_loop, set_value=self.set_value, sparse_out=self.sparse_out)