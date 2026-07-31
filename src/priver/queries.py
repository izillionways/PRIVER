from __future__ import annotations


CLASS_QUERY_PHRASES = {
    "airplane": "airplanes",
    "baseball-diamond": "baseball diamonds",
    "basketball-court": "basketball courts",
    "bridge": "bridges",
    "container": "containers",
    "container-crane": "container cranes",
    "ground-track-field": "ground track fields",
    "harbor": "harbors",
    "helicopter": "helicopters",
    "large-vehicle": "large vehicles",
    "plane": "planes",
    "roundabout": "roundabouts",
    "ship": "ships",
    "small-vehicle": "small vehicles",
    "soccer-ball-field": "soccer fields",
    "storage-tank": "storage tanks",
    "swimming-pool": "swimming pools",
    "tennis-court": "tennis courts",
    "windmill": "windmills",
}


def class_query_phrase(class_name: str) -> str:
    """Return a natural-language phrase while preserving the dataset label."""
    return CLASS_QUERY_PHRASES.get(class_name, class_name.replace("-", " "))


def format_query(template: str, class_name: str) -> str:
    """Format current templates and retain support for legacy configurations."""
    return template.format(
        class_name=class_name.replace("-", " "),
        class_phrase=class_query_phrase(class_name),
    )
