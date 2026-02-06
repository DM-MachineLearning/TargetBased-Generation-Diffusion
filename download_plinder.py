from pharmadiff.datasets.plinder_dataset import PlinderGraphDataset
from tqdm import tqdm

def download_all():
    print("Initializing Plinder Train Dataset (this triggers index download)...")
    dataset = PlinderGraphDataset(split='train')
    
    print(f"Found {len(dataset)} systems. Starting caching...")
    # Iterating through the dataset triggers the PinderSystem download for each entry
    for i in tqdm(range(len(dataset))):
        try:
            _ = dataset[i]
        except Exception as e:
            print(f"Skipping failed system {i}: {e}")

if __name__ == "__main__":
    download_all()
