import torch
from src.data import RGB_IDX
from src.models import build_model, count_parameters, INIT_STRATEGIES

torch.manual_seed(1)

fake_13 = torch.randn(4, 13, 64, 64)
fake_rgb = fake_13[:, RGB_IDX]

rgb_model = build_model(3, pretrained=True)
rgb_model.eval()

with torch.no_grad():
    rgb_out = rgb_model(fake_rgb)

assert rgb_out.shape == (4, 10)
print(f"rgb: output {tuple(rgb_out.shape)}, parameters {count_parameters(rgb_model):,}")

pretrained_rgb_filters = rgb_model.conv1.weight

# 13-band models one per init strategy
for strategy in INIT_STRATEGIES:
    torch.manual_seed(1)
    model = build_model(13, strategy, pretrained=True)
    model.eval()

    with torch.no_grad():
        out = model(fake_13)

    assert out.shape == (4, 10)
    print(f"13 bands, {strategy}: output {tuple(out.shape)}, parameters {count_parameters(model):,}")

    if strategy == "random_new":
        # b4, b3, b2 hold the pretrained red, green, blue filters
        assert torch.allclose(model.conv1.weight[:, RGB_IDX], pretrained_rgb_filters)
        model.fc.load_state_dict(rgb_model.fc.state_dict())
        only_rgb = torch.zeros(4, 13, 64, 64)
        only_rgb[:, RGB_IDX] = fake_rgb
        with torch.no_grad():
            assert torch.allclose(model(only_rgb), rgb_out, atol=1e-4)

    if strategy == "mean_rgb_rescaled":
        # b4, b3, b2 hold the pretrained filters
        assert torch.allclose(model.conv1.weight[:, RGB_IDX], pretrained_rgb_filters * 3 / 13)

        #b1 for average filter
        mean_filter = pretrained_rgb_filters.mean(dim=1) * 3 / 13
        assert torch.allclose(model.conv1.weight[:, 0], mean_filter)

    if strategy == "learned_1x1":
        # mixer picking b4, b3, b2
        model.resnet.fc.load_state_dict(rgb_model.fc.state_dict())
        with torch.no_grad():
            assert torch.allclose(model(fake_13), rgb_out, atol=1e-4)

    print(" passed")

#13 bands without pretraining
scratch = build_model(13, pretrained=False)
with torch.no_grad():
    assert scratch(fake_13).shape == (4, 10)
print("13 bands, no pretraining: passed")

print("\nall model checks passed")
