import torch


class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                        lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                        weight_decay=args.arch_weight_decay)

        self._init_arch_parameters = []
        for alpha in self.model.arch_parameters():
            alpha_init = torch.zeros_like(alpha)
            alpha_init.data.copy_(alpha)
            self._init_arch_parameters.append(alpha_init)

    def reset_arch_parameters(self):
        for alpha, alpha_init in zip(self.model.arch_parameters(), self._init_arch_parameters):
            alpha.data.copy_(alpha_init.data)

    def step(self, input_train, target_train, input_valid, target_valid, *args, **kwargs):
        shared = self._step_fo(input_train, target_train, input_valid, target_valid)
        return shared

    def _step_fo(self, input_train, target_train, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()
        self.optimizer.step()
        return None

