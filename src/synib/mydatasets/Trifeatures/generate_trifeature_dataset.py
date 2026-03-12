import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image
import tqdm

from trifeatures import Trifeatures, make_dirs

def process_task(args):
    dataset, root, s, t, c, is_test = args
    if is_test == 1:
        directory = os.path.join(root, "test")
        image_array = dataset._render_stimulus(s, c, t)
        image = Image.fromarray((image_array * 255.).astype(np.uint8))
        image.save(os.path.join(directory, "%s_%s_%s.png" % (s, t, c)))
    else:
        directory = os.path.join(root, "train")
        for i in range(dataset.num_per_combination):
            image_array = dataset._render_stimulus(s, c, t)
            image = Image.fromarray((image_array * 255.).astype(np.uint8))
            image.save(os.path.join(directory, "%s_%s_%s_%i.png" % (s, t, c, i)))

def generate_data_patched(self):
    """
    Patched generate_data method that features multithreading and a progress bar directly built in.
    """
    print("Images not found. Generating dataset with parallel processing...")
    split = self._split_array()
    
    train_directory = os.path.join(self.root, "train")
    test_directory = os.path.join(self.root, "test")
    make_dirs([train_directory, test_directory])
    
    tasks = []
    idx = 0
    for s in self.BASE_SHAPES:
        for t in self.BASE_TEXTURES:
            for c in self.BASE_COLORS.keys():
                tasks.append((self, self.root, s, t, c, split[idx]))
                idx += 1
                
    num_workers = os.cpu_count() or 8
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        for f in tqdm.tqdm(as_completed(futures), total=len(tasks), desc="Generating Images"):
            f.result()

    print("Dataset generated!")

# Monkey patch generate_data dynamically without altering trifeatures.py directly
Trifeatures.generate_data = generate_data_patched

def main():
    parser = argparse.ArgumentParser(description="Multithreaded script for Trifeatures dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the dataset")
    parser.add_argument("--num_per_combination", type=int, default=3, help="Number of views per combination")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    dataset = Trifeatures(root=args.output_dir, num_per_combination=args.num_per_combination, seed=args.seed)

if __name__ == "__main__":
    main()
