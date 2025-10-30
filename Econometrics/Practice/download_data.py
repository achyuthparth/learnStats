"""
Dataset Downloader for Wooldridge Econometrics

This script helps download datasets from the Wooldridge collection.
"""

import requests
from pathlib import Path
from typing import List
import time

# Import config
from config import DATA_DIR, DATASET_BASE_URL

# Common datasets by chapter (5th edition)
DATASETS_BY_CHAPTER = {
    1: [],
    2: ['wage1', 'bwght', 'ceosal1', 'meap01'],
    3: ['gpa1', 'wage1', 'ceosal1', 'sleep75', 'wage2', 'hprice1', 'ceosal2', 'attend'],
    4: ['wage1', 'gpa1', 'lawsch85', 'meap93', 'attend', 'mlb1', 'ceosal2'],
    5: ['wage1', 'rdchem', 'vote1', 'attend', 'hprice1'],
    6: ['wage1', 'hprice1', 'crime1', 'wage2'],
    7: ['wage1', 'gpa3', 'wage2', 'mlb1', 'crime1', 'hprice1'],
    8: ['wage1', 'crime1', 'gpa3', 'attend', 'meap93'],
    9: ['gpa1', 'wage1', 'hprice1', 'rdchem', 'crime1'],
    10: ['crime1', 'fertil3', 'hseinv', 'phillips'],
    11: ['gpa1', 'crime1', 'nyse'],
    12: ['fertil1', 'wage1', 'crime2', 'crime3', 'crime4'],
    13: ['fertil1', 'cps78_85', 'injury', 'jtrain', 'rental', 'ez'],
    14: ['kielmc', 'mroz', 'smoke', 'fertil1'],
    15: ['mroz', 'fertil2', 'crime1', 'loanapp'],
    16: ['fringe', 'jtrain', 'wagepan'],
    17: ['mroz', 'smoke', 'affairs', 'pntsprd', 'crime1'],
    18: ['mroz', 'fertil1', 'recid', 'fringe'],
    19: ['fertil1', 'gpa1', 'vote1'],
}


def download_file(url: str, save_path: Path) -> bool:
    """
    Download a file from a URL.
    
    Parameters:
    -----------
    url : str
        URL to download from
    save_path : Path
        Local path to save the file
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"  ✓ Saved to {save_path}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error: {e}")
        return False


def download_dataset(dataset_name: str, download_txt: bool = True) -> bool:
    """
    Download a Wooldridge dataset (.xls and optionally .txt files).
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (without extension)
    download_txt : bool
        Whether to also download the .txt description file
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    success = True
    
    # Download .xls file
    xls_url = f"{DATASET_BASE_URL}/{dataset_name}.xls"
    xls_path = DATA_DIR / f"{dataset_name}.xls"
    
    if xls_path.exists():
        print(f"  ⊙ {dataset_name}.xls already exists")
    else:
        success = download_file(xls_url, xls_path) and success
    
    # Download .txt file
    if download_txt:
        txt_url = f"{DATASET_BASE_URL}/{dataset_name}.txt"
        txt_path = DATA_DIR / f"{dataset_name}.txt"
        
        if txt_path.exists():
            print(f"  ⊙ {dataset_name}.txt already exists")
        else:
            download_file(txt_url, txt_path)  # Don't fail if txt is missing
    
    return success


def download_chapter_datasets(chapter: int) -> None:
    """
    Download all datasets for a specific chapter.
    
    Parameters:
    -----------
    chapter : int
        Chapter number (1-19)
    """
    if chapter not in DATASETS_BY_CHAPTER:
        print(f"Chapter {chapter} not found in dataset list.")
        return
    
    datasets = DATASETS_BY_CHAPTER[chapter]
    
    if not datasets:
        print(f"No datasets listed for Chapter {chapter}")
        return
    
    print(f"\n{'=' * 80}")
    print(f"DOWNLOADING DATASETS FOR CHAPTER {chapter}")
    print(f"{'=' * 80}\n")
    print(f"Datasets to download: {', '.join(datasets)}\n")
    
    success_count = 0
    for dataset in datasets:
        if download_dataset(dataset):
            success_count += 1
        time.sleep(0.5)  # Be nice to the server
    
    print(f"\n{'=' * 80}")
    print(f"Downloaded {success_count}/{len(datasets)} datasets successfully")
    print(f"{'=' * 80}\n")


def download_multiple_chapters(chapters: List[int]) -> None:
    """
    Download datasets for multiple chapters.
    
    Parameters:
    -----------
    chapters : List[int]
        List of chapter numbers
    """
    all_datasets = set()
    
    for chapter in chapters:
        if chapter in DATASETS_BY_CHAPTER:
            all_datasets.update(DATASETS_BY_CHAPTER[chapter])
    
    all_datasets = sorted(all_datasets)
    
    print(f"\n{'=' * 80}")
    print(f"DOWNLOADING DATASETS FOR CHAPTERS: {', '.join(map(str, chapters))}")
    print(f"{'=' * 80}\n")
    print(f"Total unique datasets: {len(all_datasets)}\n")
    
    success_count = 0
    for dataset in all_datasets:
        if download_dataset(dataset):
            success_count += 1
        time.sleep(0.5)
    
    print(f"\n{'=' * 80}")
    print(f"Downloaded {success_count}/{len(all_datasets)} datasets successfully")
    print(f"{'=' * 80}\n")


def interactive_download():
    """Interactive mode for downloading datasets."""
    print("=" * 80)
    print("WOOLDRIDGE DATASET DOWNLOADER")
    print("=" * 80)
    print("\nOptions:")
    print("1. Download datasets for a specific chapter")
    print("2. Download datasets for multiple chapters")
    print("3. Download a specific dataset by name")
    print("4. Download all common datasets (Chapters 2-10)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '1':
        chapter = int(input("Enter chapter number (1-19): "))
        download_chapter_datasets(chapter)
    
    elif choice == '2':
        chapters_str = input("Enter chapter numbers separated by commas (e.g., 2,3,4): ")
        chapters = [int(c.strip()) for c in chapters_str.split(',')]
        download_multiple_chapters(chapters)
    
    elif choice == '3':
        dataset = input("Enter dataset name (e.g., wage1): ")
        download_dataset(dataset)
    
    elif choice == '4':
        print("\nDownloading commonly used datasets (Chapters 2-10)...")
        download_multiple_chapters(list(range(2, 11)))
    
    elif choice == '5':
        print("Exiting...")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    # Make sure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    # Run interactive mode
    interactive_download()
