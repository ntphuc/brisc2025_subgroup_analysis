import torch

from src.model import build_model
from src.paper_metrics import GROUPS, hard_batch_metrics, per_sample_bce_dice_loss


def test_all_paper_architectures_output_shape():
    x = torch.randn(2, 3, 64, 64)
    for arch in ("unet_effb0", "unetpp_effb0", "deeplabv3p_resnet50"):
        model = build_model(arch, pretrained_encoder=False).eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 1, 64, 64)


def test_per_sample_loss_and_metrics_finite():
    logits = torch.randn(3, 1, 32, 32)
    target = (torch.rand(3, 1, 32, 32) > 0.8).float()
    loss = per_sample_bce_dice_loss(logits, target)
    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()
    m = hard_batch_metrics(logits, target, 0.5)
    assert m["dice"].shape == (3,)
    assert torch.isfinite(m["dice"]).all()


def test_group_count_is_nine():
    assert len(GROUPS) == 9
