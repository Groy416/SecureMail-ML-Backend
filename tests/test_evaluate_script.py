from __future__ import annotations

from scripts.evaluate_pcap_bundle import build_parser


def test_pcap_evaluation_requires_a_persisted_dataset_run() -> None:
    args = build_parser().parse_args(
        [
            "--dataset-run",
            "datasets/runs/synthetic-pcap-example",
            "--bundle",
            "models/grouped-105-capture",
        ]
    )

    assert args.dataset_run.name == "synthetic-pcap-example"
    assert args.bundle.name == "grouped-105-capture"
