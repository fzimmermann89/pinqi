import concurrent.futures
import gzip
import hashlib
import io
import pickle
import re
import urllib.request
import warnings
from pathlib import Path

import h5py
import numpy as np
import requests
from tqdm import tqdm

import argparse


def download(outdir, cachedir, workers=4):
    URL = "http://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html"
    page = requests.get(URL)
    subjects = re.findall(r"option value=(\d*)>", page.text)

    classesSignal = [
        "gry",
        "wht",
        "csf",
        "mrw",
        "dura",
        "fat",
        "fat2",
        "mus",
        "m-s",
        "ves",
    ]
    classesBG = [
        "bck",
        "skl",
    ]
    classes = classesBG + classesSignal

    def one_subject(subject, outfilename, cachedir, classes):
        def load_url(url, cachedir, timeout=60):
            h = hashlib.sha256(url.encode("utf-8")).hexdigest()
            try:
                res = pickle.load(open(Path(cachedir) / h, "rb"))
            except (FileNotFoundError, EOFError):
                with urllib.request.urlopen(url, timeout=timeout) as conn:
                    res = conn.read()
                try:
                    pickle.dump(res, open(Path(cachedir) / h, "wb"))
                except Exception as e:
                    warnings.warn(f"could not cache {str(e)}")
            return res

        def unpack(data, dtype, shape):
            return np.frombuffer(gzip.open(io.BytesIO(data)).read(), dtype=dtype).reshape(shape)

        def cut(x):
            s = np.array(x.shape)
            return x[tuple((slice(s0, -s1)) for s0, s1 in zip(s % 16 // 2, s % 16 - s % 16 // 2))]

        def norm_(values):
            for i, x in enumerate(values):
                values[i] = np.clip(x - np.min(x[50], (0, 1)), 0, 4096)
            sum_values = sum(values)
            for i, x in enumerate(values):
                x = np.divide(x, sum_values, where=sum_values != 0)
                x[sum_values == 0] = i == 0
                x = (x * (2 ** 8 - 1)).astype(np.uint8)
                values[i] = x
            return values

        values = norm_(
            [
                cut(
                    unpack(
                        load_url(
                            f"http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?do_download_alias=subject{subject}_{c}&format_value=raw_short&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D",
                            cachedir=cachedir,
                        ),
                        shape=(362, 434, 362),
                        dtype=np.uint16,
                    )
                )
                for c in classes
            ]
        )

        bg = sum(values[: len(classesBG)])
        values = np.stack(values[len(classesBG):], -1)

        with h5py.File(outfilename, "w") as f:
            f.create_dataset(
                "classes", values.shape, dtype=values.dtype, data=values, chunks=(16, 16, 16, values.shape[-1]),
            )  # ,shuffle=True, compression="lzf")
            f.create_dataset("background", bg.shape, dtype=values.dtype, data=bg, chunks=(32, 32, 32))  # , shuffle=True, compression="lzf")
            f.attrs["classnames"] = list(classesSignal)
            f.attrs["subject"] = int(subject)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one_subject, subject, Path(outdir) / f"s{subject}.h5", cachedir, classes): subject for subject in subjects}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            s = futures[future]
            try:
                fn = future.result()
            except Exception as e:
                print("%s generated an exception: %s" % (s, e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download brainweb")
    parser.add_argument("path", help="outputpath")
    parser.add_argument("--cache", help="cache, default /tmp", default="/tmp/")
    parser.add_argument("-j", help="threads", default=4, type=int)
    args = parser.parse_args()
    print("downloading...")
    download(args.path, args.cache, args.j)
    print("done!")
