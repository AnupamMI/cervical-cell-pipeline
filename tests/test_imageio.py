import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image

from ncmask.imageio import ImageLoadError, load_bgr8

RNG = np.random.default_rng(0)
RGB = RNG.integers(0, 256, (40, 60, 3), np.uint8)


@pytest.mark.parametrize("ext", [".png", ".jpg", ".bmp", ".tif"])
def test_8bit_rgb_identical_to_training_reader(tmp_path, ext):
    """For 8-bit colour files the loader must return exactly cv2.imread(path), the call the training code used."""
    p = tmp_path / f"a{ext}"
    Image.fromarray(RGB).save(p)
    img, info = load_bgr8(p)
    assert img.dtype == np.uint8 and img.shape == (40, 60, 3)
    assert np.array_equal(img, cv2.imread(str(p)))
    if ext in (".png", ".bmp", ".tif"):            # lossless: channel order must be BGR
        assert np.array_equal(img, RGB[..., ::-1])


def test_grayscale_8bit(tmp_path):
    p = tmp_path / "g.png"; Image.fromarray(RGB[..., 0]).save(p)
    img, info = load_bgr8(p)
    assert info["src_channels"] == 1 and img.shape == (40, 60, 3)
    assert np.array_equal(img, cv2.imread(str(p))) and np.array_equal(img[..., 0], RGB[..., 0])


def test_rgba_drops_alpha(tmp_path):
    p = tmp_path / "a.png"; Image.fromarray(np.dstack([RGB, np.full((40, 60), 7, np.uint8)])).save(p)
    img, _ = load_bgr8(p)
    assert np.array_equal(img, RGB[..., ::-1]) and np.array_equal(img, cv2.imread(str(p)))


def test_16bit_rgb_and_gray(tmp_path):
    v16 = RGB.astype(np.uint16) * 257 + 3
    p = tmp_path / "c16.tif"; tifffile.imwrite(p, v16)              # RGB, photometric rgb
    img, info = load_bgr8(p, "cv2")
    assert info["src_dtype"] == "uint16" and img.shape == (40, 60, 3)
    assert np.array_equal(img, cv2.imread(str(p)))                   # = the training reader
    sc, _ = load_bgr8(p, "scale")
    assert np.array_equal(sc, np.round(v16 / 257).astype(np.uint8)[..., ::-1])
    g = tmp_path / "g16.png"; cv2.imwrite(str(g), v16[..., 0])
    img, info = load_bgr8(g, "cv2")
    assert info["src_channels"] == 1 and np.array_equal(img, cv2.imread(str(g)))
    assert np.array_equal(load_bgr8(g, "scale")[0][..., 1], np.round(v16[..., 0] / 257).astype(np.uint8))
    # 12-bit data stored in 16-bit: 'max' uses the full 8-bit range
    t12 = (RGB.astype(np.uint16) * 16)
    q = tmp_path / "t12.tif"; tifffile.imwrite(q, t12)
    img, _ = load_bgr8(q, "max")
    assert img.max() == 255
    assert load_bgr8(q, "cv2")[0].max() <= 16


def test_16bit_rgba_and_gray_alpha(tmp_path):
    v = np.dstack([RGB.astype(np.uint16) * 257, np.full((40, 60), 65535, np.uint16)])
    p = tmp_path / "rgba16.png"; cv2.imwrite(str(p), v)
    img, info = load_bgr8(p, "cv2")
    assert img.shape == (40, 60, 3) and np.array_equal(img, cv2.imread(str(p)))


def test_exif_orientation_applied(tmp_path):
    p = tmp_path / "e.jpg"
    ex = Image.fromarray(RGB).getexif(); ex[0x0112] = 6               # rotate 90 cw on display
    Image.fromarray(RGB).save(p, exif=ex, quality=95)
    img, _ = load_bgr8(p)
    assert img.shape == (60, 40, 3)
    assert np.array_equal(img, cv2.imread(str(p)))


def test_corrupt_missing_and_unsupported(tmp_path):
    bad = tmp_path / "bad.png"; bad.write_bytes(b"\x89PNG\r\n\x1a\nnot really a png")
    with pytest.raises(ImageLoadError):
        load_bgr8(bad)
    trunc = tmp_path / "trunc.jpg"
    Image.fromarray(RGB).save(tmp_path / "ok.jpg")
    trunc.write_bytes((tmp_path / "ok.jpg").read_bytes()[:40])
    with pytest.raises(ImageLoadError):
        load_bgr8(trunc)
    empty = tmp_path / "empty.tif"; empty.write_bytes(b"")
    with pytest.raises(ImageLoadError):
        load_bgr8(empty)
    with pytest.raises(ImageLoadError):
        load_bgr8(tmp_path / "nope.png")
    f32 = tmp_path / "f.tif"; tifffile.imwrite(f32, RGB.astype(np.float32), photometric="rgb")
    with pytest.raises(ImageLoadError):
        load_bgr8(f32)
