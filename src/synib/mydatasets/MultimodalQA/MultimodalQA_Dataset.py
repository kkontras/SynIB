import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any, Optional

class MultimodalQA_Dataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Any] = None,
        include_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.include_types = include_types
        self.exclude_types = exclude_types
        
        self.dataset_dir = os.path.join(data_root, "dataset")
        self.images_dir = os.path.join(data_root, "final_dataset_images")
        
        # Load questions for the split
        split_file = f"MMQA_{split}.jsonl"
        self.questions = self._load_jsonl(os.path.join(self.dataset_dir, split_file))
        
        # Filter questions if needed
        if self.include_types or self.exclude_types:
            original_len = len(self.questions)
            filtered_questions = []
            for q in self.questions:
                q_type = q.get('metadata', {}).get('type')
                if self.include_types and q_type not in self.include_types:
                    continue
                if self.exclude_types and q_type in self.exclude_types:
                    continue
                filtered_questions.append(q)
            self.questions = filtered_questions
            print(f"Filtered questions. Original: {original_len}, New: {len(self.questions)}")
            if self.include_types:
                print(f"  - Included: {self.include_types}")
            if self.exclude_types:
                print(f"  - Excluded: {self.exclude_types}")
        
        # Index contexts
        print(f"Indexing MultiModalQA {split} contexts...")
        self.tables = self._index_jsonl(os.path.join(self.dataset_dir, "MMQA_tables.jsonl"))
        self.texts = self._index_jsonl(os.path.join(self.dataset_dir, "MMQA_texts.jsonl"))
        self.images_meta = self._index_jsonl(os.path.join(self.dataset_dir, "MMQA_images.jsonl"))
        
    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        data = []
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            return []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _index_jsonl(self, path: str) -> Dict[str, Dict[str, Any]]:
        index = {}
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            return index
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                index[item['id']] = item
        return index

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx: int):
        q = self.questions[idx]
        metadata = q.get('metadata', {})
        
        supporting_context = q.get('supporting_context', [])
        
        # Retrieve context from metadata
        table_id = metadata.get('table_id')
        table_raw = self.tables.get(table_id) if table_id else None
        table_dict = {table_id: table_raw} if table_raw else {}
        # Extract the inner table data if it exists, otherwise use the raw object
        table = table_raw.get('table') if table_raw and 'table' in table_raw else table_raw
        
        text_ids = set(metadata.get('text_doc_ids', []))
        image_ids = set(metadata.get('image_doc_ids', []))
        
        # Also add anything explicitly mentioned in supporting_context
        for ctx in supporting_context:
            if ctx['doc_part'] == 'text':
                text_ids.add(ctx['doc_id'])
            elif ctx['doc_part'] == 'image':
                image_ids.add(ctx['doc_id'])
            elif ctx['doc_part'] == 'table':
                if ctx['doc_id'] not in table_dict and ctx['doc_id'] in self.tables:
                    table_dict[ctx['doc_id']] = self.tables[ctx['doc_id']]

        texts = []
        text_dict = {}
        for tid in text_ids:
            if tid in self.texts:
                item = self.texts[tid]
                texts.append(item)
                text_dict[tid] = item
        
        images = []
        image_dict = {}
        for img_id in image_ids:
            if img_id in self.images_meta:
                meta = self.images_meta[img_id]
                img_path = os.path.join(self.images_dir, meta['path'])
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        if self.transform:
                            img = self.transform(img)
                        images.append(img)
                        image_dict[img_id] = img
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
        
        return {
            "qid": q['qid'],
            "question": q['question'],
            "answers": q['answers'],
            "table": table,
            "table_metadata": table_raw, 
            "table_dict": table_dict,
            "texts": texts,
            "text_dict": text_dict,
            "images": images,
            "image_dict": image_dict,
            "metadata": metadata,
            "supporting_context": supporting_context
        }

if __name__ == "__main__":
    # Quick test
    root = "/esat/hagalaz/tpoporda/Projects/SynIB/src/synib/mydatasets/MultimodalQA"
    ds = MultimodalQA_Dataset(root, split="train")
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample Question: {sample['question']}")
        print(f"Sample Answer: {sample['answers']}")
        print(f"Number of images: {len(sample['images'])}")
