from priver.dota import build_class_queries as build_dota_queries
from priver.queries import class_query_phrase, format_query
from priver.soda import build_class_queries as build_soda_queries


TEMPLATES = [
    "Locate image patches containing {class_phrase}.",
    "Find regions that show {class_phrase}.",
    "Retrieve local areas containing {class_phrase}.",
    "Which image regions contain {class_phrase}?",
]


def _sample_rows() -> tuple[list[dict], list[dict]]:
    images = [{"image_id": "P0001", "image_path": "/tmp/P0001.png"}]
    objects = [
        {
            "object_id": "P0001_0000",
            "image_id": "P0001",
            "class_name": "small-vehicle",
        }
    ]
    return images, objects


def test_class_query_phrase_uses_natural_plural_forms() -> None:
    assert class_query_phrase("small-vehicle") == "small vehicles"
    assert class_query_phrase("tennis-court") == "tennis courts"
    assert class_query_phrase("soccer-ball-field") == "soccer fields"


def test_format_query_supports_current_and_legacy_placeholders() -> None:
    assert (
        format_query("Locate patches containing {class_phrase}.", "storage-tank")
        == "Locate patches containing storage tanks."
    )
    assert (
        format_query("Locate a {class_name}.", "storage-tank")
        == "Locate a storage tank."
    )


def test_dataset_builders_create_stable_four_template_queries() -> None:
    images, objects = _sample_rows()
    for builder in (build_dota_queries, build_soda_queries):
        queries = builder(images, objects, TEMPLATES)
        assert [row["query_id"] for row in queries] == [
            "P0001_small-vehicle_t0",
            "P0001_small-vehicle_t1",
            "P0001_small-vehicle_t2",
            "P0001_small-vehicle_t3",
        ]
        assert [row["text"] for row in queries] == [
            "Locate image patches containing small vehicles.",
            "Find regions that show small vehicles.",
            "Retrieve local areas containing small vehicles.",
            "Which image regions contain small vehicles?",
        ]
