#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
import os

# Limit threads to prevent resource exhaustion
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1" 

from p_tqdm import p_map
import imagehash
from PIL import Image
import itertools
from typing import Dict, List
from collections import Counter

def generate_image_hash(image_path: Path) -> str:
    try:
        img = Image.open(image_path)
        angles = (0, 90, -90, 180)
        hashes = [str(imagehash.phash(img.rotate(angle))) for angle in angles]
        return "-".join(sorted(hashes))
    except Exception as e:
        logging.error("Error processing %s: %s", image_path, e)
        return ""

def generate_hashes_for_dataset(dataset_folder: Path, exts=None, workers=8) -> dict[str, str]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png"}
    images_dir = dataset_folder / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        logging.warning("Images directory does not exist: %s", images_dir)
        return {}
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    results = p_map(generate_image_hash, image_paths, num_cpus=workers)
    return {p.name: h for p, h in zip(image_paths, results) if h}

def analyze_annotations(dataset_folder: Path, split: str) -> Dict:
    annotation_file = dataset_folder / split / "annotations" / "annotation.json"
    if not annotation_file.exists():
        logging.warning("Annotation file does not exist: %s", annotation_file)
        return {}
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    num_images = len(coco_data.get('images', []))
    num_annotations = len(coco_data.get('annotations', []))
    
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    category_counts = Counter()
    for ann in coco_data.get('annotations', []):
        category_counts[ann.get('category_id', 0)] += 1
    
    image_id_to_annotations = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann.get('image_id')
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)
    
    annotations_per_image = [len(anns) for anns in image_id_to_annotations.values()]
    avg_annotations_per_image = sum(annotations_per_image) / len(annotations_per_image) if annotations_per_image else 0
    max_annotations_per_image = max(annotations_per_image) if annotations_per_image else 0
    
    return {
        'num_images': num_images,
        'num_annotations': num_annotations,
        'categories': categories,
        'category_counts': category_counts,
        'avg_annotations_per_image': avg_annotations_per_image,
        'max_annotations_per_image': max_annotations_per_image,
        'images_with_annotations': len(image_id_to_annotations),
        'images_without_annotations': num_images - len(image_id_to_annotations)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Generate image hashes and/or analyze existing hash results."
    )
    parser.add_argument(
        "dataset_folder", type=Path,
        help="Path to the root of the dataset"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="List of dataset splits to process"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory where hash files are saved"
    )
    parser.add_argument(
        "--analyze", "-a", action="store_true",
        help="Analyze the dataset: performs image hash analysis and annotation statistics"
    )
    parser.add_argument(
        "--prefix", "-p", type=str, default=None,
        help="Prefix for hash filenames (defaults to dataset folder name)"
    )
    parser.add_argument(
        "--workers", type=int, default=16, 
        help="Number of parallel workers for processing (default: 8)"
    )
    parser.add_argument(
        "--skip-hash-generation", action="store_true",
        help="Skip hash generation and only perform analysis on existing hash files"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or args.dataset_folder.name

    if not args.skip_hash_generation:
        for split in args.splits:
            split_folder = args.dataset_folder / split
            logging.info("Processing split '%s' in %s", split, split_folder)
            hashes = generate_hashes_for_dataset(split_folder, workers=args.workers)
            out_file = args.output_dir / f"{prefix}_{split}_hashes.json"
            out_file.write_text(json.dumps(hashes, indent=2))
            logging.info("Saved %d hashes to %s", len(hashes), out_file)

    if args.analyze:
        logging.info("Running dataset analyses...")
        
        hash_data = {
            split: load_hashes_file(
                args.output_dir / f"{prefix}_{split}_hashes.json"
            ) for split in args.splits
        }
        
        unique_hash_counts = {
            split: len(set(hashes.values())) 
            for split, hashes in hash_data.items() if hashes
        }
        
        analyze_dataset_annotations(args.dataset_folder, args.splits, unique_hash_counts)
        analyze_hashes(args.output_dir, prefix, args.splits, summary_only=True)

def load_hashes_file(file_path: Path) -> Dict[str, str]:
    if not file_path.exists():
        logging.warning("Hash file not found: %s", file_path)
        return {}
    return json.load(file_path.open())

def analyze_dataset_annotations(dataset_folder: Path, splits: List[str], unique_hash_counts: Dict[str, int] = None) -> None:
    dataset_name = dataset_folder.name
    print(f"\n=== Annotation Statistics for '{dataset_name}' ===")
    
    annotation_stats = {}
    for split in splits:
        print(f"\nAnalyzing annotations for '{split}' split:")
        stats = analyze_annotations(dataset_folder, split)
        annotation_stats[split] = stats
        
        if not stats:
            print(f"  No annotations found for {split}")
            continue
            
        print(f"  Images: {stats['num_images']}")
        print(f"  Buildings (annotations): {stats['num_annotations']}")
        print(f"  Images with annotations: {stats['images_with_annotations']}")
        print(f"  Images without annotations: {stats['images_without_annotations']}")
        print(f"  Average buildings per image: {stats['avg_annotations_per_image']:.2f}")
        print(f"  Maximum buildings in a single image: {stats['max_annotations_per_image']}")
        
        if stats['category_counts']:
            print("  Annotations per category:")
            for cat_id, count in stats['category_counts'].items():
                cat_name = stats['categories'].get(cat_id, f"Unknown ({cat_id})")
                print(f"    {cat_name}: {count}")
    
    if len(splits) > 1:
        print("\nComparison across splits:")
        
        if unique_hash_counts:
            values = {split: unique_hash_counts.get(split, 0) for split in splits if unique_hash_counts.get(split, 0) > 0}
            if values:
                print(f"  num_unique_hashes:", end=" ")
                print(", ".join(f"{split}: {value}" for split, value in values.items()))
        
        for stat in ['num_images', 'num_annotations', 'avg_annotations_per_image']:
            values = {split: stats.get(stat, 0) for split, stats in annotation_stats.items() if stats}
            if values:
                print(f"  {stat}:", end=" ")
                print(", ".join(f"{split}: {value:.2f}" if isinstance(value, float) else f"{split}: {value}" 
                               for split, value in values.items()))

def analyze_hashes(results_dir: Path, prefix: str, splits: List[str], summary_only: bool = False) -> None:
    mappings: Dict[str, Dict[str, str]] = {
        split: load_hashes_file(
            results_dir / f"{prefix}_{split}_hashes.json"
        ) for split in splits
    }

    unique_sets: Dict[str, set] = {
        split: set(m.values()) for split, m in mappings.items()
    }

    if summary_only:
        print(f"\n=== Image Hash Statistics for '{prefix}' ===")
    
    for split, m in mappings.items():
        print(f"{split}: {len(m)} images, {len(unique_sets[split])} unique hashes")

    inv: Dict[str, Dict[str, List[str]]] = {split: {} for split in splits}
    for split, m in mappings.items():
        for img, hv in m.items():
            inv[split].setdefault(hv, []).append(img)
            
    internal_duplicates = {}
    for split in splits:
        duplicate_hashes = [h for h, img_list in inv[split].items() if len(img_list) > 1]
        internal_duplicates[split] = duplicate_hashes
        
        duplicate_count = len(mappings[split]) - len(unique_sets[split])
        if duplicate_count > 0:
            print(f"\n{split} has {duplicate_count} duplicate images ({len(duplicate_hashes)} hash groups):")
            for idx, h in enumerate(sorted(duplicate_hashes)):
                if idx < 10:
                    print(f"  Group {idx+1}: Hash {h} appears in {len(inv[split][h])} images:")
                    print(f"    {', '.join(inv[split][h])}")
                elif idx == 10:
                    print(f"  ... and {len(duplicate_hashes) - 10} more duplicate hash groups")
                    break

    overlap_stats = {}
    overlap_images = {}
    for s1, s2 in itertools.combinations(splits, 2):
        ov = unique_sets[s1].intersection(unique_sets[s2])
        p1 = len(ov) / (len(unique_sets[s1]) or 1) * 100
        p2 = len(ov) / (len(unique_sets[s2]) or 1) * 100
        overlap_stats[(s1, s2)] = {
            'overlap_count': len(ov),
            'percent_s1': p1,
            'percent_s2': p2
        }
        
        if len(ov) > 0:
            overlap_images[(s1, s2)] = [(inv[s1][h], inv[s2][h]) for h in ov]
        else:
            overlap_images[(s1, s2)] = []
        
        if not summary_only:
            print(
                f"{s1}-{s2} overlap: {len(ov)} hashes ({p1:.2f}% of {s1}, {p2:.2f}% of {s2})"
            )
            if ov:
                print(f"Overlapping images between {s1} and {s2}:")
                for hv in sorted(ov):
                    print(
                        f"  Hash {hv}: {s1} -> {inv[s1].get(hv, [])}; {s2} -> {inv[s2].get(hv, [])}"
                    )

    triple_overlap_images = []
    if len(splits) == 3:
        all_ov = set.intersection(*unique_sets.values())
        triple_overlap_count = len(all_ov)
        
        if triple_overlap_count > 0:
            for hv in all_ov:
                triple_overlap_images.append(
                    {split: inv[split].get(hv, []) for split in splits}
                )
        
        if not summary_only:
            print(f"Overlap across all three splits: {triple_overlap_count} hashes")
            if all_ov:
                print("Overlapping images across all three splits:")
                for hv in sorted(all_ov):
                    imgs_dict = {split: inv[split].get(hv, []) for split in splits}
                    print(f"  Hash {hv}: {imgs_dict}")
    
    if len(splits) > 1:
        print(f"\n=== Image Hash Overlap Summary for '{prefix}' ===")
        print("Pairwise overlaps:")
        for (s1, s2), stats in overlap_stats.items():
            print(f"  {s1}-{s2}: {stats['overlap_count']} unique hashes " +
                  f"({stats['percent_s1']:.2f}% of {s1}, {stats['percent_s2']:.2f}% of {s2})")
            
            if stats['overlap_count'] > 0:
                print(f"    Overlapping images:")
                max_to_show = min(10, len(overlap_images[(s1, s2)]))
                for i, (imgs1, imgs2) in enumerate(overlap_images[(s1, s2)][:max_to_show]):
                    print(f"      {i+1}. {s1}: {', '.join(imgs1)} ⟷ {s2}: {', '.join(imgs2)}")
                
                if len(overlap_images[(s1, s2)]) > max_to_show:
                    print(f"      ... and {len(overlap_images[(s1, s2)]) - max_to_show} more")
        
        if len(splits) == 3:
            s1, s2, s3 = splits
            p1 = triple_overlap_count / (len(unique_sets[s1]) or 1) * 100
            p2 = triple_overlap_count / (len(unique_sets[s2]) or 1) * 100
            p3 = triple_overlap_count / (len(unique_sets[s3]) or 1) * 100
            print(f"Triple overlap ({s1}-{s2}-{s3}): {triple_overlap_count} unique hashes " +
                  f"({p1:.2f}% of {s1}, {p2:.2f}% of {s2}, {p3:.2f}% of {s3})")
            
            if triple_overlap_count > 0:
                print(f"    Overlapping images:")
                max_to_show = min(10, len(triple_overlap_images))
                for i, imgs_dict in enumerate(triple_overlap_images[:max_to_show]):
                    print(f"      {i+1}. {s1}: {', '.join(imgs_dict[s1])} ⟷ " +
                          f"{s2}: {', '.join(imgs_dict[s2])} ⟷ {s3}: {', '.join(imgs_dict[s3])}")
                
                if len(triple_overlap_images) > max_to_show:
                    print(f"      ... and {len(triple_overlap_images) - max_to_show} more")

if __name__ == "__main__":
    main()

# Usage examples:
# Basic usage (generate hashes and save to results/):
#   python analyse_dataset.py /path/to/dataset
#
# Generate hashes for specific splits:
#   python analyse_dataset.py /path/to/dataset --splits train val
#
# Generate hashes and analyze dataset:
#   python analyse_dataset.py /path/to/dataset --analyze
#
# Skip hash generation and only analyze existing hash files:
#   python analyse_dataset.py /path/to/dataset --analyze --skip-hash-generation
#
# Use custom output directory and prefix:
#   python analyse_dataset.py /path/to/dataset --output-dir custom_results --prefix dataset_name        