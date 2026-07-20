import inspect

from script.experiment.fpct_confirmatory_runner import ARM_ORDER, training_config


def test_arm_order_is_position_balanced():
    assert set(ARM_ORDER) == set(range(45, 57))
    for arm in ("c_pre", "c_post", "f"):
        assert [order.count(arm) for order in ARM_ORDER.values()] == [1] * 12
        assert [sum(order[position] == arm for order in ARM_ORDER.values()) for position in range(3)] == [4, 4, 4]


def test_formal_recipe_is_exact_64_step_contract(tmp_path):
    lock = {"run_uid": "test", "assets": {"training_alignment_sidecar_2048": {"container_path": "/fpct-assets/train2048.pt"}}}
    config = training_config(lock, 45, "f", tmp_path)
    assert config["data"]["kwargs"]["num_samples"] == 2048
    assert config["data"]["train_ratio"] == 1.0
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["training"]["per_device_train_batch_size"] == 1
    assert config["training"]["num_processes"] == 2
    assert config["training"]["expected_optimizer_steps"] == 64
    assert config["model"]["fpct_alignment_sanitizer"] == "certified_slot0_v1"
    assert config["model"]["include_response"] is False


def test_triplet_runner_never_selectively_retries_an_arm():
    source = inspect.getsource(__import__("script.experiment.fpct_confirmatory_runner", fromlist=["train_triplet"]).train_triplet)
    assert "retry" not in source.lower()


def test_activation_floor_comes_from_operator_null_controls():
    module = __import__("script.experiment.fpct_confirmatory_runner", fromlist=["gpu_numerical"])
    source = inspect.getsource(module.gpu_numerical)
    assert "replicated_collapse_output_delta" in source
    assert "m1_output_delta" in source
    floor_source = source[source.index("activation_floor"):source.index("serializable")]
    assert "row_sum_error" not in floor_source
