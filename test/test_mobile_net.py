import pytest
import torch

from cars.models.mobile_nets import MobileNetV1, MobileNetV2, SmallMobileNetV3, LargeMobileNetV3
from cars.config import mobile2_params, mobile3_small, mobile3_large

input_tensor = None


@pytest.fixture(autouse=True)
def mock_input_tensor():
    global input_tensor
    input_tensor = torch.zeros(1, 3, 224, 224)


def test_if_mobile_v1_forwards_tensor():
    model = MobileNetV1(100, 1)
    assert model(input_tensor).size() == torch.Size([1, 100])


def test_if_mobile_v1_with_scaling_forwards_tensor():
    model = MobileNetV1(100, 0.7)
    assert model(input_tensor).size() == torch.Size([1, 100])


def test_if_mobile_v2_forwards_tensor():
    model = MobileNetV2(mobile2_params, 100, 1)
    assert model(input_tensor).size() == torch.Size([1, 100])


def test_if_mobile_v2_with_scaling_forwards_tensor():
    model = MobileNetV2(mobile2_params, 100, 0.4)
    assert model(input_tensor).size() == torch.Size([1, 100])


# def test_if_mobile_v3_small_forwards_tensor():
#     model = SmallMobileNetV3(mobile3_small, "small", 100, 1)
#     assert model(input_tensor).size() == torch.Size([1, 100])
#
# def test_if_mobile_v3_small_with_scaling_forwards_tensor():
#     model = MobileNetV3(mobile3_small, "small", 100, 0.75)
#     assert model(input_tensor).size() == torch.Size([1, 100])
#
#
# def test_if_mobile_v3_large_forwards_tensor():
#     model = MobileNetV3(mobile3_large, "large", 100, 1)
#     assert model(input_tensor).size() == torch.Size([1, 100])
#
#
# def test_if_mobile_v3_large_with_scaling_forwards_tensor():
#     model = MobileNetV3(mobile3_large, "large", 100, 0.85)
#     assert model(input_tensor).size() == torch.Size([1, 100])
