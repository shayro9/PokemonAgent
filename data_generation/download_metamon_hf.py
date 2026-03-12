"""
download_metamon_hf.py
======================
Download metamon parsed replay datasets directly from HuggingFace.

This is MUCH simpler than installing metamon - just uses huggingface_hub to download
the pre-parsed replay files.

INSTALL
-------
pip install huggingface_hub

USAGE
-----
# Download Gen 9 OU replays
python download_metamon_hf.py --format gen9ou --output-dir data/metamon

# Download multiple formats
python download_metamon_hf.py --format gen1ou gen9ou --output-dir data/metamon

# List available formats
python download_metamon_hf.py --list

DATASET INFO
------------
Repository: jakegrigsby/metamon-parsed-replays
Format: Each format is a .tar.gz file containing JSON files
Structure: {format}.tar.gz -> {format}/*.json

Each JSON file is a parsed battle trajectory with:
- observations: Battle state from player POV
- actions: Action choices (already mapped to indices!)
- rewards: Rewards for each step
- dones: Episode termination flags
- metadata: Battle info, ratings, etc.

Available Formats:
- gen1ou, gen1uu, gen1nu, gen1ubers
- gen2ou, gen2uu, gen2nu, gen2ubers
- gen3ou, gen3uu, gen3nu, gen3ubers
- gen4ou, gen4uu, gen4nu, gen4ubers
- gen9ou
"""

import argparse
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

# All available formats in the metamon dataset
AVAILABLE_FORMATS = [
    # Gen 1
    "gen1ou", "gen1uu", "gen1nu", "gen1ubers",
    # Gen 2
    "gen2ou", "gen2uu", "gen2nu", "gen2ubers",
    # Gen 3
    "gen3ou", "gen3uu", "gen3nu", "gen3ubers",
    # Gen 4
    "gen4ou", "gen4uu", "gen4nu", "gen4ubers",
    # Gen 9
    "gen9ou",
]

REPO_ID = "jakegrigsby/metamon-parsed-replays"


def download_format(format_name: str, output_dir: Path, extract: bool = True):
    """Download a single format's dataset from HuggingFace.
    
    Args:
        format_name: Battle format (e.g., "gen9ou")
        output_dir: Directory to save the data
        extract: If True, extract the tar.gz file
    """
    if format_name not in AVAILABLE_FORMATS:
        print(f"ERROR: Unknown format '{format_name}'")
        print(f"Available formats: {', '.join(AVAILABLE_FORMATS)}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the tar.gz file
    filename = f"{format_name}.tar.gz"
    print(f"\nDownloading {filename} from HuggingFace...")
    print(f"  Repository: {REPO_ID}")
    print(f"  This may take a while (files are large)...")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ Downloaded to: {downloaded_path}")
        
        # Extract if requested
        if extract:
            extract_path = output_dir / format_name
            if extract_path.exists():
                print(f"  ⚠ Directory {extract_path} already exists, skipping extraction")
            else:
                print(f"  Extracting to: {extract_path}")
                with tarfile.open(downloaded_path, 'r:gz') as tar:
                    tar.extractall(path=output_dir)
                print(f"  ✓ Extracted successfully")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error downloading {format_name}: {e}")
        return False


def list_formats():
    """Print all available formats."""
    print("\nAvailable Formats:")
    print("=" * 50)
    
    gens = {
        1: [f for f in AVAILABLE_FORMATS if f.startswith("gen1")],
        2: [f for f in AVAILABLE_FORMATS if f.startswith("gen2")],
        3: [f for f in AVAILABLE_FORMATS if f.startswith("gen3")],
        4: [f for f in AVAILABLE_FORMATS if f.startswith("gen4")],
        9: [f for f in AVAILABLE_FORMATS if f.startswith("gen9")],
    }
    
    for gen, formats in gens.items():
        print(f"\nGen {gen}:")
        for fmt in formats:
            print(f"  - {fmt}")
    
    print(f"\nTotal: {len(AVAILABLE_FORMATS)} formats")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Download metamon parsed replay datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--format",
        nargs="+",
        help="Battle format(s) to download (e.g., gen9ou gen1ou)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/metamon"),
        help="Directory to save downloaded files (default: data/metamon)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract tar.gz files (just download)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available formats and exit",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_formats()
        return
    
    if not args.format:
        print("ERROR: Must specify --format or use --list")
        parser.print_help()
        return
    
    # Download each requested format
    print(f"\nDownloading to: {args.output_dir.resolve()}")
    success_count = 0
    
    for format_name in args.format:
        if download_format(format_name, args.output_dir, extract=not args.no_extract):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Downloaded {success_count}/{len(args.format)} formats successfully")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
