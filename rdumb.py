# copied from https://github.com/oripress/CCC
import math
from copy import deepcopy

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F


class RDumb(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.num_samples_update_1 = (
            0  # number of samples after First filtering, exclude unreliable samples
        )
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = math.log(1000) * 0.4  # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = (
            0.05  # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)
        )

        self.current_model_probs = (
            None  # the moving average of probability vector (Eqn. 4)
        )

        params, param_names = collect_params(model)
        model = configure_model(model)

        self.model = model
        self.optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)

        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )
        self.total_steps = 0

    def forward(
            self,
            x,
    ):
        if self.total_steps % 1000 == 0:
            load_model_and_optimizer(
                self.model, self.optimizer, self.model_state, self.optimizer_state
            )
            self.current_model_probs = None

        # forward
        outputs = self.model(x)
        # adapt
        entropys = softmax_entropy(outputs)
        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys = entropys[filter_ids_1]
        self.ent = entropys.size(0)
        # filter redundant samples
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(
                self.current_model_probs.unsqueeze(dim=0),
                outputs[filter_ids_1].softmax(1).detach(),
                dim=1,
            )
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            self.div = filter_ids_2[0].size(0)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = self.update_model_probs(
                self.current_model_probs,
                outputs[filter_ids_1][filter_ids_2].softmax(1).detach(),
            )
        else:
            updated_probs = self.update_model_probs(
                self.current_model_probs, outputs[filter_ids_1].softmax(1)
            )
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)

        if x[ids1][ids2].size(0) != 0:
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        self.num_samples_update_2 += entropys.size(0)
        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.current_model_probs = updated_probs
        self.total_steps += 1

        return outputs

    def update_model_probs(self, current_model_probs, new_probs):
        if current_model_probs is None:
            if new_probs.size(0) == 0:
                return None
            else:
                with torch.no_grad():
                    return new_probs.mean(0)
        else:
            if new_probs.size(0) == 0:
                with torch.no_grad():
                    return current_model_probs
            else:
                with torch.no_grad():
                    return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"]:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def erase_bn_stats(model):
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model
