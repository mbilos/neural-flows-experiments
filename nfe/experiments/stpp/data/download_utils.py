# Copyright (c) Facebook, Inc. and its affiliates.

import os

from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parents[2] / 'data/stpp'


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """
    import urllib

    if not filename:
        filename = os.path.basename(url)
    fpath = DATA_DIR / root
    fpath.mkdir(parents=True, exist_ok=True)
    fpath = fpath / filename

    urllib.request.urlretrieve(url, str(fpath), reporthook=gen_bar_updater())

    return True
