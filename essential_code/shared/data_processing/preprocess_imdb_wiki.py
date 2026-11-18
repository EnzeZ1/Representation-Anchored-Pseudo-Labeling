"""Preprocess IMDB-WIKI dataset: extract (image_path, age) pairs."""

import json
import numpy as np

from pathlib import Path
from datetime import datetime


def load_mat_metadata(mat_path):
    """Load metadata from IMDB-WIKI .mat file."""
    from scipy.io import loadmat
    meta = loadmat(mat_path, squeeze_me=True)

    # Navigate nested structure
    for key in meta:
        if key.startswith('__'):
            continue
        data = meta[key]
        break

    # Extract fields — handle both flat and nested .mat formats
    if hasattr(data, 'dtype') and data.dtype.names:
        # Structured array
        fields = {name: data[name].item() if data[name].ndim == 0 else data[name]
                  for name in data.dtype.names}
    else:
        # Try nested
        fields = {}
        for name in ['dob', 'photo_taken', 'full_path', 'face_score', 'second_face_score', 'gender']:
            try:
                fields[name] = data[0, 0][name].flatten()
            except:
                pass

    return fields


def matlab_datenum_to_year(datenum):
    """Convert Matlab datenum to birth year."""
    try:
        if datenum < 1 or np.isnan(datenum):
            return None
        # Matlab datenum: days since Jan 0, 0000
        dt = datetime.fromordinal(int(datenum) - 366)
        return dt.year
    except:
        return None


def process_dataset(root, name, mat_name):
    """Process one dataset (imdb or wiki)."""
    root = Path(root)
    mat_path = root / name / f'{mat_name}.mat'

    if not mat_path.exists():
        print(f'  {mat_path} not found, skipping')
        return []

    print(f'  Loading {mat_path}...')
    fields = load_mat_metadata(str(mat_path))

    paths = fields.get('full_path', [])
    dobs = fields.get('dob', [])
    photo_taken = fields.get('photo_taken', [])
    face_scores = fields.get('face_score', [])
    second_face = fields.get('second_face_score', [])

    if len(paths) == 0:
        print(f'  No data found in {mat_path}')
        return []

    items = []
    skipped = {'bad_age': 0, 'bad_face': 0, 'second_face': 0, 'missing': 0}

    for i in range(len(paths)):
        # Get image path
        p = paths[i]
        if isinstance(p, np.ndarray):
            p = str(p.flat[0])
        img_path = root / name / str(p)

        if not img_path.exists():
            skipped['missing'] += 1
            continue

        # Compute age
        birth_year = matlab_datenum_to_year(float(dobs[i]))
        if birth_year is None:
            skipped['bad_age'] += 1
            continue

        age = int(photo_taken[i]) - birth_year
        if age < 0 or age > 100:
            skipped['bad_age'] += 1
            continue

        # Filter bad face detections
        if len(face_scores) > i:
            fs = face_scores[i]
            if np.isnan(fs) or fs < 1.0:
                skipped['bad_face'] += 1
                continue

        # Filter images with second face
        if len(second_face) > i:
            sf = second_face[i]
            if not np.isnan(sf):
                skipped['second_face'] += 1
                continue

        items.append({'path': str(img_path), 'age': int(age)})

    print(f'  {name}: {len(items)} valid, skipped: {skipped}')
    return items


def main():
    root = Path('data/imdb_wiki')

    all_items = []

    # Process IMDB
    print('Processing IMDB...')
    imdb_items = process_dataset(root, 'imdb_crop', 'imdb')
    all_items.extend(imdb_items)

    # Process Wiki
    print('Processing Wiki...')
    wiki_items = process_dataset(root, 'wiki_crop', 'wiki')
    all_items.extend(wiki_items)

    print(f'\nTotal valid images: {len(all_items)}')

    if all_items:
        ages = [item['age'] for item in all_items]
        print(f'Age range: [{min(ages)}, {max(ages)}]')
        print(f'Mean age: {np.mean(ages):.1f}, Std: {np.std(ages):.1f}')

        # Save
        out_path = root / 'metadata.json'
        with open(out_path, 'w') as f:
            json.dump(all_items, f)
        print(f'Saved to {out_path}')

        # Also save a simple CSV for quick inspection
        csv_path = root / 'metadata.csv'
        with open(csv_path, 'w') as f:
            f.write('path,age\n')
            for item in all_items:
                f.write(f"{item['path']},{item['age']}\n")
        print(f'Saved CSV to {csv_path}')


if __name__ == '__main__':
    main()
