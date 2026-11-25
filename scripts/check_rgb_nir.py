"""Checks for the RGB+NIR (4-band) option. Run from the project root:
    python -m scripts.check_rgb_nir
"""
import torch

from src.data import RGB_IDX, RGB_NIR_IDX, get_dataset, select_bands
from src.models import build_model, count_parameters, INIT_STRATEGIES

# ---------- data ----------
assert RGB_NIR_IDX == [3, 2, 1, 7], RGB_NIR_IDX

fake = torch.randn(13, 64, 64)
assert torch.equal(select_bands(fake, "rgb_nir"), fake[[3, 2, 1, 7]])

nir_ds = get_dataset("train", "rgb_nir", 50, 1)
all_ds = get_dataset("train", "all", 50, 1)
assert len(nir_ds) == len(all_ds) == 500
for i in [0, 1, 250, 499]:
    nir_img, nir_label = nir_ds[i]
    all_img, all_label = all_ds[i]
    assert nir_img.shape == (4, 64, 64)
    assert nir_label == all_label
    assert torch.allclose(nir_img, all_img[RGB_NIR_IDX])
print("data: rgb_nir gives (4, 64, 64), same images/labels/normalization as 'all': passed")

# ---------- models ----------
torch.manual_seed(1)
fake_4 = torch.randn(4, 4, 64, 64)
fake_rgb = fake_4[:, :3]

rgb_model = build_model(3, pretrained=True)
rgb_model.eval()
with torch.no_grad():
    rgb_out = rgb_model(fake_rgb)
pretrained = rgb_model.conv1.weight

for strategy in INIT_STRATEGIES:
    torch.manual_seed(1)
    model = build_model(4, strategy, pretrained=True)
    model.eval()
    with torch.no_grad():
        out = model(fake_4)
    assert out.shape == (4, 10)
    print(f"4 bands, {strategy}: output {tuple(out.shape)}, parameters {count_parameters(model):,}")

    if strategy == "random_new":
        assert torch.allclose(model.conv1.weight[:, :3], pretrained)
        model.fc.load_state_dict(rgb_model.fc.state_dict())
        only_rgb = fake_4.clone()
        only_rgb[:, 3] = 0
        with torch.no_grad():
            assert torch.allclose(model(only_rgb), rgb_out, atol=1e-4)

    if strategy == "mean_rgb_rescaled":
        assert torch.allclose(model.conv1.weight[:, :3], pretrained * 3 / 4)
        assert torch.allclose(model.conv1.weight[:, 3], pretrained.mean(dim=1) * 3 / 4)

    if strategy == "learned_1x1":
        model.resnet.fc.load_state_dict(rgb_model.fc.state_dict())
        with torch.no_grad():
            assert torch.allclose(model(fake_4), rgb_out, atol=1e-4)
    print("  passed")

scratch = build_model(4, pretrained=False)
with torch.no_grad():
    assert scratch(fake_4).shape == (4, 10)
print("4 bands, no pretraining: passed")

print("\nall rgb_nir checks passed")
