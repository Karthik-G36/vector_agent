"""Remove image background using rembg (U2Net model)."""
from rembg import remove


def remove_background(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as f:
        result = remove(f.read())
    with open(output_path, "wb") as f:
        f.write(result)
    print(f"      Background removed -> {output_path}")
