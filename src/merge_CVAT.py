import glob
import os
import xml.etree.ElementTree as ET


def find_annotation_files(root_dir: str = ".") -> list[str]:
    pattern = os.path.join(root_dir, "dataset_*", "**", "*.xml")
    files = glob.glob(pattern, recursive=True)
    files.sort()
    return files


def merge_annotations(root_dir: str = ".", output_file: str = "full_annotation.xml") -> None:
    xml_files = find_annotation_files(root_dir)

    if not xml_files:
        print(f"[!] Не знайдено жодного XML-файлу за шляхом: {root_dir}/dataset_*/")
        return

    print(f"[+] Знайдено {len(xml_files)} файл(ів):")
    for f in xml_files:
        print(f"    {f}")

    base_version = None
    base_meta = None
    seen_names: dict[str, tuple[int, str]] = {}
    all_images: list[ET.Element] = []

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as e:
            print(f"[!] Помилка парсингу {xml_path}: {e} — пропускаємо")
            continue

        root = tree.getroot()

        if base_version is None:
            base_version = root.find("version")
        if base_meta is None:
            base_meta = root.find("meta")

        added = 0
        replaced = 0
        skipped = 0
        no_boxes = 0

        for img in root.findall("image"):
            name = img.get("name", "")
            has_boxes = len(img.findall("box")) > 0

            if not has_boxes:
                no_boxes += 1
                continue

            if name not in seen_names:
                seen_names[name] = (len(all_images), xml_path)
                all_images.append(img)
                added += 1
            else:
                existing_idx, existing_src = seen_names[name]
                existing_has_boxes = len(all_images[existing_idx].findall("box")) > 0

                if has_boxes and not existing_has_boxes:
                    all_images[existing_idx] = img
                    seen_names[name] = (existing_idx, xml_path)
                    replaced += 1
                    print(f"    [↑] Дублікат '{name}': замінено ({existing_src}) → ({xml_path})")
                else:
                    reason = "обидва мають box-и" if has_boxes else "обидва без box-ів"
                    print(f"    [~] Дублікат '{name}' з {xml_path} ({reason}) — залишаємо з {existing_src}")
                    skipped += 1

        print(f"    {xml_path}: додано {added}, замінено {replaced}, пропущено {skipped}, без box-ів {no_boxes}")

    if not all_images:
        print("[!] Немає зображень для об'єднання.")
        return

    for new_id, img in enumerate(all_images):
        img.set("id", str(new_id))

    merged_root = ET.Element("annotations")
    if base_version is not None:
        merged_root.append(base_version)
    if base_meta is not None:
        merged_root.append(base_meta)
    for img in all_images:
        merged_root.append(img)

    try:
        ET.indent(merged_root, space="  ")
    except AttributeError:
        pass

    ET.ElementTree(merged_root).write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"\n[✓] Збережено {len(all_images)} зображень → {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="full_annotation.xml")
    args = parser.parse_args()

    merge_annotations(root_dir=args.root, output_file=args.output)