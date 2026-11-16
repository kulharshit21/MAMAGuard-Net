from mamaguardnet.cluster.gpu import evaluate_gpus, parse_gpu_csv

SAMPLE = """1, NVIDIA A100-SXM4-40GB, GPU-a, 40960, 10265, 30695, 11
5, NVIDIA A100-SXM4-40GB, GPU-b, 40960, 0, 40960, 0
6, NVIDIA A100-SXM4-40GB, GPU-c, 40960, 0, 40960, 0
7, NVIDIA A100-SXM4-40GB, GPU-d, 40960, 0, 40960, 0
"""


def test_user_snapshot_does_not_pass_four_gpu_threshold():
    report = evaluate_gpus(parse_gpu_csv(SAMPLE), [1, 5, 6, 7], min_free_mib=36000, max_utilization=10)
    assert not report["safe"]
    assert [row["index"] for row in report["unsafe"]] == [1]


def test_all_four_pass_when_free():
    report = evaluate_gpus(parse_gpu_csv(SAMPLE.replace("10265, 30695, 11", "0, 40960, 0")), [1, 5, 6, 7], min_free_mib=36000, max_utilization=10)
    assert report["safe"]

