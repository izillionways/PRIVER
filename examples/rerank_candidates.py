from priver.reranking import (
    InterScaleSupportSpec,
    SameScaleSupportSpec,
    rerank_candidates,
)


candidates = [
    {
        "patch_id": "large",
        "score": 0.76,
        "bbox": [0, 0, 1024, 1024],
        "patch_size": 1024,
    },
    {
        "patch_id": "small-supported",
        "score": 0.72,
        "bbox": [128, 128, 640, 640],
        "patch_size": 512,
    },
    {
        "patch_id": "small-isolated",
        "score": 0.74,
        "bbox": [1400, 1400, 1912, 1912],
        "patch_size": 512,
    },
]

reranked = rerank_candidates(
    candidates,
    SameScaleSupportSpec(),
    InterScaleSupportSpec(),
    same_scale_weight=0.25,
    inter_scale_weight=1.0,
)

for rank, candidate in enumerate(reranked, start=1):
    print(rank, candidate["patch_id"], f"{candidate['priver_score']:.3f}")
