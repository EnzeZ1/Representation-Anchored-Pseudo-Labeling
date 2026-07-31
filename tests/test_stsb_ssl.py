import torch
from torch import nn

from data_processing.stsb_ssl import STSBUnlabeledViews
from models.hpl_uncertainty import UncertaintyLearner
from training.stsb_ssl import hpl_meta_step


def test_unlabeled_views_do_not_expose_targets():
    cohort={"records":[{"sentence1":"A person is not running.","sentence2":"Someone moves.","stable_id":"x","score":4.0}]}
    dataset=STSBUnlabeledViews(cohort,[0],0)
    item=dataset[0]
    assert "target" not in item and item["weak_sentence1"]=="A person is not running."


def test_hpl_uncertainty_shape_and_gradients():
    learner=UncertaintyLearner()
    output=learner(torch.randn(4,2))
    assert output.shape==(4,1)
    output.mean().backward()
    assert all(parameter.grad is not None for parameter in learner.parameters())


def test_hpl_bilevel_step_updates_uncertainty():
    class TinyPairModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(3, 4)
            self.dropout = nn.Dropout(0.0)
            self.head = nn.Linear(4, 1)

        def forward_features(self, input_ids, attention_mask):
            return self.backbone(input_ids.float())

    model = TinyPairModel()
    learner = UncertaintyLearner()
    head_optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    uncertainty_optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)
    batch = {
        "input_ids": torch.randn(5, 3), "attention_mask": torch.ones(5, 3),
        "target": torch.randn(5),
    }
    before = [parameter.detach().clone() for parameter in learner.parameters()]
    value = hpl_meta_step(
        model, learner, head_optimizer, uncertainty_optimizer,
        batch, batch, batch, batch, "roberta_base", 1.0,
    )
    assert torch.isfinite(torch.tensor(value))
    assert any(not torch.equal(old, new) for old, new in zip(before, learner.parameters()))
