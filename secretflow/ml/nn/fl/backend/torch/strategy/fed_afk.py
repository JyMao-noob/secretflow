import copy
from typing import Tuple

import numpy as np
import torch
from secretflow.ml.nn.core.torch import BuilderType
from secretflow.ml.nn.fl.backend.torch.fl_base import BaseTorchModel
from secretflow.ml.nn.fl.strategy_dispatcher import register_strategy


class FedAFK(BaseTorchModel):
    """
    A simple implementation of FedAFK.
    FedAFK 除了维护在服务器客户端间通信的全局模型, 还会在各客户端处维护一个本地模型, 推理的时候使用本地模型.
    FedAFK 包含了三种简单但很有效的设计:
    >   模型解耦: 将深度学习模型划分为特征提取器(feature extractor, a.k.a: body)和分类头(classifier head, a.k.a: head)
    >   自适应特征聚合: 使用一个可学习的参数 mu 来聚合全局特征提取器和本地特征提取器
    >   知识迁移: 将全局模型的知识迁移至本地模型(通过计算 L2 norm)
    """

    def __init__(
            self,
            builder_base: BuilderType,
            random_seed: int = None,
            skip_bn: bool = False,
            **kwargs,
    ):

        super().__init__(builder_base, random_seed, skip_bn, **kwargs)
        # 在 FedAFK 实现中，显示地定义了 global model，用以区分 local model，在通信时，传递 global model 参数
        self.g_model = copy.deepcopy(self.model)
        # 初始化参数
        self.mu = kwargs.get('mu', 0.5)
        self.lamb = kwargs.get('lamb', 0.3)

    def train_step(
            self,
            weights: np.ndarray,
            cur_steps: int,
            train_steps: int,
            **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Accept ps model params, then do local train

        Args:
            weights: global weight from params server
            cur_steps: current train step
            train_steps: local training steps
            kwargs: strategy-specific parameters
        Returns:
            Parameters after local training
        """

        assert self.model is not None, "Model cannot be none, please give model define"

        if weights is not None:
            # 从 weights 中恢复出 global model
            self.g_model.update_weights(weights)

        self.g_model.train()
        self.model.train()
        refresh_data = kwargs.get("refresh_data", False)
        if refresh_data:
            self._reset_data_iter()
        num_sample = 0
        dp_strategy = kwargs.get('dp_strategy', None)
        logs = {}

        # 模型解耦 g->global  p->personalized(local)
        g_all_layers = list(self.g_model.children())
        g_body_layers = g_all_layers[:-1]
        g_head_layer = g_all_layers[-1]
        g_params = [param for layer in g_body_layers for param in layer.parameters()]

        self.g_optimizer = torch.optim.SGD(self.g_model.parameters(), lr=0.005)

        p_all_layers = list(self.model.children())
        p_body_layers = p_all_layers[:-1]
        p_head_layer = p_all_layers[-1]
        p_params = [param for layer in p_body_layers for param in layer.parameters()]

        # 对于本地模型, 首先固定住 head, 只更新 body
        for layer in p_body_layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in p_head_layer.parameters():
            param.requires_grad = False

        self.p_body_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.005)

        for step in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]

            # 通过监督损失 SGD 梯度更新整个 global model
            g_loss_erm = self.g_model.training_step((x, y), step + cur_steps, sample_weight=s_w)
            self.g_optimizer.zero_grad()
            g_loss_erm.backward()
            self.g_optimizer.step()

            # 通过知识传递损失+监督损失 SGD 梯度更新 local model's feature extractor
            p_loss_erm = self.model.training_step((x, y), step + cur_steps, sample_weight=s_w)
            p_loss_kt = 0.0
            for p, pp in zip(g_params, p_params):
                p_loss_kt += torch.sum((p - pp) ** 2)  # 计算 L2 norm
            p_loss = (1 - self.lamb) * p_loss_erm + self.lamb * p_loss_kt
            self.p_body_optimizer.zero_grad()
            p_loss.backward()
            self.p_body_optimizer.step()

            # 自适应更新权重参数 mu
            grad_mu = 0
            for g, p in zip(g_params, p_params):
                dif = p.data - g.data
                grad = self.mu * p.grad.data + (1 - self.mu) * g.grad.data
                grad_mu += dif.view(-1).T.dot(grad.view(-1))
            grad_mu += 0.02 * self.mu
            self.mu = self.mu - 0.005 * grad_mu
            self.mu = np.clip(self.mu.item(), 0.0, 1.0)

            # 利用 mu 更新特征提取器
            # personalized feature extractor = mu * local feature extractor + (1 - mu) * global feature extractor
            for p, g in zip(p_params, g_params):
                p.data = self.mu * p + (1 - self.mu) * g

        # 对于本地模型, 再固定住 body, 更新 head
        for layer in p_body_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in p_head_layer.parameters():
            param.requires_grad = True

        self.p_head_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.005)

        self._reset_data_iter()
        for step in range(train_steps):
            x, y, s_w = self.next_batch()
            num_sample += x.shape[0]
            # 通过监督损失 SGD 梯度更新 local model's classifier head
            loss = self.model.training_step((x, y), step + cur_steps, sample_weight=s_w)
            self.p_head_optimizer.zero_grad()
            loss.backward()
            self.p_head_optimizer.step()

        loss_value = p_loss.item()
        logs['train-loss'] = loss_value

        self.logs = self.transform_metrics(logs)
        self.wrapped_metrics.extend(self.wrap_local_metrics())
        self.epoch_logs = copy.deepcopy(self.logs)

        # 与 server 通信的只有 global model
        model_weights = self.g_model.get_weights(return_numpy=True)

        # DP operation
        if dp_strategy is not None:
            if dp_strategy.model_gdp is not None:
                model_weights = dp_strategy.model_gdp(model_weights)
        return model_weights, num_sample

    def apply_weights(self, weights, **kwargs):
        """Accept ps model params,then update local model

        Args:
            weights: global weight from params server
        """
        if weights is not None:
            self.g_model.update_weights(weights)


@register_strategy(strategy_name='fed_afk', backend='torch')
class PYUFedAFK(FedAFK):
    pass
