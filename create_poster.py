import argparse
import os
import pickle
from glob import glob
from functools import partial
from random import shuffle

import torch
import torchvision
import imageio
import numpy as np
import skimage.measure
from skimage.transform import resize
from tqdm import tqdm


def _load_image(imp):
    images = []
    im = imageio.imread(imp)
    positions = [
        [
            [174, 12],
            [174, 260],
            [174, 508],
        ],
        [
            [422, 12],
            [422, 260],
            [422, 508]
        ],
        [
            [670, 12],
            [670, 260],
            [670, 508]
        ]
    ]
    s1 = [413, 251]
    im_shape = tuple(ss1 - ss0 for ss0, ss1 in zip(positions[0][0], s1))
    for i in range(3):
        for j in range(3):
            pos = positions[i][j]
            crop = tuple(
                slice(po, po + sh) for po, sh in zip(pos, im_shape)
            )
            this_im = im[crop]
            images.append(this_im)
    return images


def _load_images(pattern):
    image_paths = glob(pattern)
    images = []
    for imp in image_paths:
        this_images = _load_image(imp)
        images.extend(this_images)
    return images


def random_loading(folder, ):
    pattern = os.path.join(folder, "*.png")
    images = _load_images(pattern)
    shuffle(images)
    return images


# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
def _normalize(im):
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    im = (im - mean) / std
    return im.astype("float32")


def _compute_features(model, images, fnames, features):
    with torch.no_grad():
        for name, im in tqdm(zip(fnames, images), total=len(fnames), desc="Compute image features"):
            if name in features:
                continue
            else:
                input_ = im[..., :3].transpose((2, 0, 1)).astype("float32")
                input_ = _normalize(input_)
                input_ = torch.from_numpy(input_[None])
                feats = model(input_)
                features[name] = feats
    return features


def perceptual_loading(reference_path=None):
    pattern = "../dall-e/*.png"
    images = _load_images(pattern)
    fnames = [os.path.basename(path) for path in glob(pattern)]
    fnames = [f"{name}_{i}" for name in fnames for i in range(9)]
    assert len(fnames) == len(images)

    model = torchvision.models.vgg19(pretrained=True).features[:30]
    tmp_path = "./features.pickle"
    if os.path.exists(tmp_path):
        with open(tmp_path, "rb") as f:
            features = pickle.load(f)
    else:
        features = {}

    features = _compute_features(model, images, fnames, features)
    with open(tmp_path, "wb") as f:
        pickle.dump(features, f)

    # just take a random image as reference
    if reference_path is None:
        sample_id = np.random.randint(0, len(features))
        ref_feats = list(features.values())[sample_id]
    else:  # compute reference features from the image path
        ref_images = _load_image(reference_path)
        ref_features = {}
        ref_features = _compute_features(model, ref_images, np.arange(len(ref_images)), ref_features)
        ref_features = [ff.numpy() if not isinstance(ff, np.ndarray) else ff
                        for ff in ref_features.values()]
        ref_features = np.concatenate(ref_features, axis=0)
        ref_feats = np.mean(ref_features, axis=0)[None]

    distances = []
    for feat in features.values():
        try:
            ref_feats = ref_feats.numpy() if not isinstance(ref_feats, np.ndarray) else ref_feats
            feat = feat.numpy() if not isinstance(feat, np.ndarray) else feat
            distances.append(np.mean(
                np.square(ref_feats - feat)
            ))
        except ValueError:
            distances.append(np.inf)

    im_ids = np.argsort(distances)
    images = [images[im_id] for im_id in im_ids]

    return images


def _get_letters_and_features(im, model):

    def feature_extractor(image):
        im_resized = np.zeros((3, 32, 32), dtype="float32")
        for chan, im in enumerate(image):
            im_resized[chan] = resize(im, (32, 32))
        features = model(torch.from_numpy(im_resized[None]))
        features = features.numpy().squeeze()
        return features.ravel()

    p0 = [125, 17]
    p1 = [158, 658]
    crop = tuple(slice(p00, p11) for p00, p11 in zip(p0, p1))

    im = im[crop][..., :3]
    im_g = np.mean(im, axis=-1)
    im = _normalize(im.transpose((2, 0, 1)))

    mask = im_g > 100
    letter_masks = skimage.measure.label(mask)
    props = skimage.measure.regionprops(letter_masks)
    feats = {}

    xminmax = {}
    with torch.no_grad():
        for prop in props:
            size = prop.area
            if size < 7:  # i-dot
                continue
            bb = prop.bbox
            min_x, max_x = bb[1], bb[3]
            crop = np.s_[:, bb[0]:bb[2], bb[1]:bb[3]]
            letter_im = im[crop]
            feats[min_x] = [prop.label] + feature_extractor(letter_im)
            xminmax[min_x] = (min_x, max_x)

    feats = [feats[k] for k in sorted(feats.keys())]
    labels = [ff[0] for ff in feats]
    feats = [ff[1:] for ff in feats]
    xminmax = [xminmax[k] for k in sorted(xminmax)]

    word_stops = []
    new_letters = np.zeros_like(letter_masks)
    for i, letter in enumerate(labels):
        new_letters[letter_masks == letter] = (i + 1)
        if i < len(xminmax) - 1:
            letter_dist = xminmax[i + 1][0] - xminmax[i][1]
            if letter_dist > 4:
                word_stops.append(i)

    return im, new_letters, np.array(feats), word_stops


def _load_letter_model():
    net = torchvision.models.vgg16(pretrained=True)
    return net.features


def fix_word(word, words):
    letter_pairs = [
        ["a", "e"], ["i", "l"], ["n", "h"]
    ]
    for (l1, l2) in letter_pairs:
        new_word = word.replace(l1, l2)
        if words.check(new_word):
            return new_word
        new_word = word.replace(l2, l1)
        if words.check(new_word):
            return new_word
    # TODO try to fix higher orders
    return new_word


def _extract_caption(path, model, dict_letters, dict_feats, words):
    im = imageio.imread(path)
    im, letter_masks, feats, stops = _get_letters_and_features(im, model)

    caption = []
    current_word = ""
    for ii, feat in enumerate(feats):
        dist = np.sum(np.square(dict_feats - feat[None]), axis=1)
        current_word += dict_letters[np.argmin(dist)]
        if ii in stops:
            caption.append(current_word)
            current_word = ""
    caption.append(current_word)

    fixed_caption = []
    for word in caption:
        if words.check(word):
            fixed_caption.append(word)
            continue
        word = fix_word(word, words)
        fixed_caption.append(word)

    return fixed_caption


def check_ocr():
    import enchant
    path = "/home/pape/Desktop/art/dall-e/craiyon_2022-6-20_14-10-11.png"
    model = _load_letter_model()
    dict_letters, dict_feats = create_dictionary(model)
    words = enchant.Dict()
    caption = _extract_caption(path, model, dict_letters, dict_feats, words)
    print(caption)


def create_dictionary(model):
    path = "/home/pape/Desktop/art/dall-e/abc.png"
    im = imageio.imread(path)
    im, letter_masks, feats, _ = _get_letters_and_features(im, model)
    assert len(feats) == 28, f"{len(feats)}"
    feats = feats[:-1]
    letters = "abcdefghijklmnopqrstuvwxyzf"
    assert len(letters) == len(feats), f"{len(letters)} != {len(feats)}"
    return letters, feats


def keyword_loading(folder, keywords, shuffle_images=True, require_all_kwds=False):
    import enchant

    pattern = os.path.join(folder, "*.png")
    images = _load_images(pattern)
    paths = glob(pattern)
    fnames = [f"{os.path.basename(path)}_{i}" for path in paths for i in range(9)]
    assert len(fnames) == len(images)

    tmp_path = "./captions.pickle"
    if os.path.exists(tmp_path):
        with open(tmp_path, "rb") as f:
            captions = pickle.load(f)
    else:
        captions = {}

    model = _load_letter_model()
    words = enchant.Dict()
    dict_letters, dict_feats = create_dictionary(model)
    for path in tqdm(paths, desc="Extract captions"):
        if path in captions:
            continue
        caption = _extract_caption(path, model, dict_letters, dict_feats, words)
        captions[path] = caption
    with open(tmp_path, "wb") as f:
        pickle.dump(captions, f)

    captions = [caption for caption in captions.values() for i in range(9)]
    assert len(captions) == len(images)
    if require_all_kwds:
        images = [im for im, cap in zip(images, captions)
                  if all(kw in cap for kw in keywords)]
    else:
        images = [im for im, cap in zip(images, captions)
                  if any(kw in cap for kw in keywords)]

    if shuffle_images:
        shuffle(images)
    return images


def create_poster(folder, output, size_inches, dpi, load_images=None):
    size_pix = tuple(siz * dpi for siz in size_inches)
    poster = np.zeros(size_pix + (4,), dtype="uint8")
    color = [255, 255, 255, 255]
    poster[..., :] = color

    if load_images is None:
        load_images = random_loading
    images = load_images(folder)
    im_shape = images[0].shape

    ims_per_dim = [psh // ish for psh, ish in zip(poster.shape[:-1], im_shape[:-1])]
    margin = [psh % ish for psh, ish in zip(poster.shape[:-1], im_shape[:-1])]
    offset = [marg // 2 for marg in margin]

    n_images_poster = np.prod(ims_per_dim)
    print("Showing", n_images_poster, "/", len(images), "images on the poster")
    images = images[:n_images_poster]

    for ii, im in enumerate(images):
        if im.shape != im_shape:
            continue
        i, j = np.unravel_index([ii], ims_per_dim)
        i, j = i[0], j[0]
        crop = tuple(
            slice(off + ind * sh, off + (ind + 1) * sh)
            for ind, sh, off in zip((i, j), im_shape[:-1], offset)
        )
        poster[crop] = im

    poster = poster[..., :3]
    if output:
        imageio.imwrite(output, poster)
    else:
        import napari
        v = napari.Viewer()
        v.add_image(poster)
        napari.run()


def main():
    size_inches = (35, 24)  # inches
    dpi = 300  # dots / inch

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="The folder with dallemini screenshots")
    parser.add_argument("--output", "-o", help="How to save the poster")
    parser.add_argument("--size", "-s", type=int, nargs="+", default=size_inches, help="The poster size in inches")
    parser.add_argument("--dpi", "-d", default=dpi)
    parser.add_argument("--keywords", "-k", type=str, nargs="+", default=[])

    # this is not working correctly yet
    # perceptual loaders
    # loader = perceptual_loading()
    # ref_path = "/home/pape/Desktop/art/dall-e/craiyon_2022-6-20_14-10-11.png"
    # loader = partial(perceptual_loading, ref_path)

    args = parser.parse_args()
    folder = args.input
    output = args.output
    size = tuple(args.size)
    assert len(size) == 2
    dpi = args.dpi

    if args.keywords:
        # ocr + keyword based loader
        loader = partial(keyword_loading, keywords=args.keywords)
        create_poster(folder, output, size, dpi, loader)
    else:
        # load all images in random order
        create_poster(folder, output, size, dpi)


if __name__ == "__main__":
    main()
    # check_ocr()
