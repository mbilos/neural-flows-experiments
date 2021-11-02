import torch
import torch.nn as nn
import torch.nn.functional as F
import stribor as st

from torchdiffeq import odeint_adjoint as odeint

from nfe.models import CouplingFlow, ResNetFlow, ContinuousGRULayer, ContinuousLSTMLayer

class MarkedTPP(nn.Module):
    def __init__(self, tpp_model, n_classes, hidden_dim):
        super().__init__()
        self.n_classes = n_classes
        self.tpp = tpp_model
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, times, marks, mask):
        time_loss, hidden = self.tpp(times, marks, mask)
        logits = self.proj(torch.cat([hidden, times], -1))
        mark_loss = F.cross_entropy(logits.view(-1, self.n_classes), marks.view(-1), reduction='none')
        mark_loss = (mark_loss * mask.view(-1)).sum() / mask.sum()
        return time_loss, mark_loss


class Diffeq(nn.Module):
    def __init__(self, dim, hidden_dims, activation, final_activation):
        super().__init__()
        self.net = st.net.MLP(dim + 1, hidden_dims, dim, activation, final_activation)
        self.intensity = st.net.MLP(dim, [], 1, final_activation='Softplus')

    def forward(self, t, state):
        """ Input: t: (), state: tuple(x (..., n, d), diff (..., n, 1)) """
        hidden, integral, diff = state
        d_integral = self.intensity(hidden) * diff
        hidden = torch.cat([t * diff, hidden], -1)
        d_hidden = self.net(hidden) * diff
        return d_hidden, d_integral, torch.zeros_like(diff).to(diff)

class JumpODE(nn.Module):
    def __init__(self, args, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = args.hidden_dim
        self.solver = args.solver
        self.atol = args.atol
        self.rtol = args.rtol
        self.options = None if args.solver == 'dopri5' else { 'step_size': args.solver_step }

        self.ode = Diffeq(args.hidden_dim, [args.hidden_dim] * args.hidden_layers,
                          args.activation, args.final_activation)
        self.embedding = nn.Embedding(n_classes, args.hidden_dim)
        self.rnn = nn.LSTMCell(1 + args.hidden_dim, args.hidden_dim)

    def forward(self, times, marks, mask):
        h = torch.zeros(times.shape[0], self.hidden_dim).to(times)
        c = torch.zeros(times.shape[0], self.hidden_dim).to(times)

        marks = self.embedding(marks.squeeze(-1))

        loss, hidden = [], []
        for i in range(times.shape[1]):
            t = times[:,i].unsqueeze(1)

            initial = (h.unsqueeze(1), torch.zeros_like(t).to(t), t)
            solution = odeint(self.ode, initial, torch.Tensor([0, 1]), method=self.solver,
                              options=self.options, atol=self.atol, rtol=self.rtol)

            h, integral, _ = (x[-1] for x in solution)
            intensity = self.ode.intensity(h)

            hidden.append(h)

            h, c = self.rnn(torch.cat([t.squeeze(1), marks[:,1]], -1), (h.squeeze(1), c))

            nll = -torch.log(intensity) + integral
            loss.append(nll)

        hidden = torch.stack(hidden, 1).squeeze(2)

        loss = torch.cat(loss, 1)
        loss = (loss * mask).sum() / mask.sum()

        return loss, hidden


class JumpFlow(nn.Module):
    def __init__(self, args, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = args.hidden_dim
        self.embedding = nn.Embedding(n_classes, args.hidden_dim)

        if args.flow_model == 'coupling':
            flow = CouplingFlow
        elif args.flow_model == 'resnet':
            flow = ResNetFlow
        else:
            raise NotImplementedError

        self.flow = flow(args.hidden_dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                         args.time_net, args.time_hidden_dim)
        self.lstm = nn.LSTMCell(1 + args.hidden_dim, args.hidden_dim)
        self.intensity = st.net.MLP(args.hidden_dim, [], 1, final_activation='Softplus')

    def forward(self, times, marks, mask):
        h = torch.zeros(times.shape[0], self.hidden_dim).to(times)
        c = torch.zeros(times.shape[0], self.hidden_dim).to(times)

        marks = self.embedding(marks.squeeze(-1))

        loss, hidden = [], []
        for i in range(times.shape[1]):
            t = times[:,i].unsqueeze(1)

            mc_samples = 30 if self.training else 100
            time_samples = torch.rand(1, mc_samples, 1).to(t) * t
            path = self.flow(h.unsqueeze(1), time_samples)
            integral = self.intensity(path).mean(1, keepdim=True) * t

            h = self.flow(h.unsqueeze(1), t)
            intensity = self.intensity(h)

            hidden.append(h)

            h, c = self.lstm(torch.cat([t.squeeze(1), marks[:,1]], -1), (h.squeeze(1), c))

            nll = -torch.log(intensity) + integral
            loss.append(nll)

        hidden = torch.stack(hidden, 1).squeeze(2)

        loss = torch.cat(loss, 1)
        loss = (loss * mask).sum() / mask.sum()

        return loss, hidden


class LogNormalMixture(nn.Module):
    def __init__(self, hidden_dim, components):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nn = st.net.MLP(hidden_dim, [hidden_dim], components * 3)

    def forward(self, h, t):
        log_weight, mu, sigma = self.nn(h).chunk(3, dim=-1)
        sigma = F.softplus(sigma)
        log_weight = F.log_softmax(log_weight, -1)

        dist = torch.distributions.LogNormal(mu, sigma)
        log_prob = torch.logsumexp(log_weight + dist.log_prob(t + 1e-8), -1, keepdim=True)
        return log_prob


class MixtureTPP(nn.Module):
    def __init__(self, args, n_classes):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.embedding = nn.Embedding(n_classes, args.hidden_dim)
        self.enc = nn.GRU(1 + args.hidden_dim, args.hidden_dim, batch_first=True)
        self.log_prob = LogNormalMixture(args.hidden_dim, args.components)

    def forward(self, times, marks, mask):
        hidden = torch.zeros(1, 1, self.hidden_dim).repeat(1, times.shape[0], 1).to(times)

        times_padded = torch.cat([torch.zeros(times.shape[0], 1, 1).to(times), times], 1)
        marks_padded = torch.cat([torch.zeros(marks.shape[0], 1, 1).to(marks), marks], 1)
        marks_emb = self.embedding(marks_padded.squeeze(-1))
        input = torch.cat([times_padded, marks_emb], -1)
        out, _ = self.enc(input, hidden)
        out = out[:,:-1]

        log_prob = self.log_prob(out, times)
        loss = -(log_prob * mask).sum() / mask.sum()
        return loss, out


class MixtureFlowTPP(nn.Module):
    def __init__(self, args, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = args.hidden_dim
        self.embedding = nn.Embedding(n_classes, args.hidden_dim)
        self.log_prob = LogNormalMixture(args.hidden_dim, args.components)

        if args.rnn == 'gru':
            rnn = ContinuousGRULayer
        elif args.rnn == 'lstm':
            rnn = ContinuousLSTMLayer
        else:
            raise NotImplementedError

        self.enc = rnn(1 + args.hidden_dim,
                       hidden_dim=args.hidden_dim,
                       model=args.model,
                       flow_model=args.flow_model,
                       hidden_layers=args.hidden_layers,
                       activation=args.activation,
                       final_activation=args.final_activation,
                       flow_layers=args.flow_layers,
                       time_net=args.time_net,
                       time_hidden_dim=args.time_hidden_dim,
                       solver=args.solver,
                       solver_step=args.solver_step)

    def forward(self, times, marks, mask):
        times = torch.cat([torch.zeros(times.shape[0], 1, 1).to(times), times], 1)
        marks = torch.cat([torch.zeros(marks.shape[0], 1).to(marks), marks.squeeze(-1)], 1)
        marks = self.embedding(marks)

        h = self.enc(torch.cat([times, marks], -1), times)

        times = times[:,1:]
        h = h[:,:-1]

        log_prob = self.log_prob(h, times)
        loss = -(log_prob * mask).sum() / mask.sum()
        return loss, h
