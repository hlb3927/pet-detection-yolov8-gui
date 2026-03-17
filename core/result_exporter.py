import json
from pathlib import Path
from datetime import datetime


def export_result_to_json(result: dict, save_dir="demo_outputs/json"):
    """
    将检测结果导出为 JSON 文件
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image_name = result.get("image_name", "result")
    stem = Path(image_name).stem

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = save_dir / f"{stem}_{timestamp}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return str(json_path)