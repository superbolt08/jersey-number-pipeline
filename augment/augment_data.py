def occlusion(img, scale, **_):
    scale = _get_param(scale, img, 0.25)
    img_np = np.array(img).copy()

    h, w = img_np.shape[:2]
    occ_w = int(w * np.random.uniform(0.1, 0.3))
    occ_h = int(h * np.random.uniform(0.2, 0.6))

    x = np.random.randint(0, w - occ_w)
    y = np.random.randint(0, h - occ_h)

    img_np[y:y+occ_h, x:x+occ_w] = np.random.randint(0, 255)
    return Image.fromarray(img_np)

def color_jitter(img, level, **_):
    factor = 1.0 + (level / 10.0) #Sets the scale of jitter between 1 and 2
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0, factor))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.0, factor))
    if random.random() < 0.8:
        img = ImageEnhance.Color(img).enhance(random.uniform(1.0, factor))

    return img

def perspective_warp(img, scale, **_):
    scale = _get_param(scale, img, 0.15)

    key = 'perspective_' + str(scale)
    op = _get_op(key, lambda: iaa.PerspectiveTransform(scale=(0.01, scale / 100.0)))

    return Image.fromarray(op(image=np.asarray(img)))

def elastic_distortion(img, alpha, **_):
    alpha = _get_param(alpha, img, 0.10)

    key = 'elastic_' + str(alpha)
    op = _get_op(key, lambda: iaa.ElasticTransformation(
        alpha=alpha,
        sigma=alpha * 0.5
    ))

    return Image.fromarray(op(image=np.asarray(img)))