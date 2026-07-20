import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_freezes_balanced_design_and_firewalls():
    manifest = json.loads((ROOT / "recipe/eval_recipe/fpct_confirmatory/confirmatory_manifest.json").read_text())
    training = manifest["formal_training"]
    assert training["seeds"] == list(range(45, 57))
    assert training["optimizer_steps"] == 64
    assert training["official_checkpoint_step"] == 64
    counts = Counter()
    for order in training["arm_order"].values():
        assert sorted(order) == ["c_post", "c_pre", "f"]
        counts.update((arm, index) for index, arm in enumerate(order))
    assert set(counts.values()) == {4}
    assert manifest["scientific_contract"]["headline_contrast"] == "f-c_post"
    assert manifest["statistics"]["sign_flip_null"] == "sharp_symmetric_sign_exchangeability"


def test_k8s_templates_are_digest_only_and_have_scoped_labels():
    directory = ROOT / "recipe/k8s/fpct_confirmatory"
    for path in directory.glob("*.yaml"):
        payload = yaml.safe_load(path.read_text())
        text = path.read_text()
        assert "__IMAGE_DIGEST__" in text
        assert "/home/lijunsi/miniconda" not in text
        assert "/home/lijunsi/projects/Cache" not in text
        assert "persistentVolumeClaim" not in text
        assert "/opt/conda/bin/python3.11" in text
        if "hostPath:" in text:
            assert "/netdisk/lijunsi/" in text
        labels = payload["metadata"]["labels"]
        assert labels["project"] == "fpct"
        assert labels["study"] == "confirmatory"
        assert labels["run_uid"] == "__RUN_UID__"
        assert payload["spec"]["backoffLimit"] == 0
