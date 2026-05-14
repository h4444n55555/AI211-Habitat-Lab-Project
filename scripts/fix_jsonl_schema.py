#!/usr/bin/env python
import json
import os

root = 'task 3/data_large'
for split in ['train', 'val']:
    path = os.path.join(root, split, 'samples.jsonl')
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ep = str(r.get('episode_id', ''))
            step = int(r.get('step_idx', 0))
            rows.append({
                'image': f"{ep}/frame_{step:03d}.png",
                'instruction': str(r['instruction']),
                'action': int(r.get('action', r.get('target_action_id'))),
            })
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f"Rewrote {split}: {len(rows)} samples")
