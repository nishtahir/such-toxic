import torch

from such_toxic import util


def test_1d_shape() -> None:
    a = [1, 2, 3, 4, 5]
    assert util.shape(a) == list(torch.tensor(a).shape)


def test_2d_shape() -> None:
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert util.shape(a) == list(torch.tensor(a).shape)


def test_3d_shape() -> None:
    a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert util.shape(a) == list(torch.tensor(a).shape)


def test_unsqueeze() -> None:
    a = [1, 2, 3]
    assert util.unsqueeze(a) == torch.tensor(a).unsqueeze(0).tolist()


def test_unsqueeze_axis_1() -> None:
    a = [1, 2, 3]
    assert util.unsqueeze(a, axis=1) == torch.tensor(a).unsqueeze(1).tolist()


def test_unsqueeze_axis_2() -> None:
    a = [[1, 2, 3]]
    assert util.unsqueeze(a, axis=2) == torch.tensor(a).unsqueeze(2).tolist()


def test_expand() -> None:
    a = [[[1], [2], [3]]]
    assert util.expand(a, [1, 3, 6]) == torch.tensor(a).expand([1, 3, 6]).tolist()


def test_mul() -> None:
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    assert util.mat_mul(a, b) == (torch.tensor(a) * torch.tensor(b)).tolist()


def test_3d_mul() -> None:
    a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    b = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

    assert util.mat_mul(a, b) == (torch.tensor(a) * torch.tensor(b)).tolist()


# def test_sum() -> None:
#     a = [1, 2, 3, 4, 5]
#     assert util.mat_sum(a) == torch.tensor(a).sum().item()


def test_sum_2d() -> None:
    a = [[1, 2, 3], [4, 5, 6]]
    assert util.mat_sum(a) == torch.tensor(a).sum().item()


def test_sum_3d() -> None:
    a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert util.mat_sum(a) == torch.tensor(a).sum().item()


def test_sum_over_dim() -> None:
    a = [[1, 2, 3], [4, 5, 6]]
    assert util.mat_sum(a, 1) == torch.sum(torch.tensor(a), 1).tolist()


def test_sum_over_dim_3d() -> None:
    a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    print(torch.tensor(a).shape)
    print(torch.tensor(a)[1, 1])
    print(torch.sum(torch.tensor(a), 2).shape)
    assert util.mat_sum(a, 1) == torch.sum(torch.tensor(a), 1).tolist()


# def test_clamp() -> None:
#     a = [[1, 2, 3], [4, 5, 6]]
#     assert util.clamp(a, 2, 4) == torch.clamp(torch.tensor(a), min=2, max=4).tolist()


# def test_3d_clamp() -> None:
#     a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
#     assert util.clamp(a, 2, 4) == torch.clamp(torch.tensor(a), min=2, max=4).tolist()
