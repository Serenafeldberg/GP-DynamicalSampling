"""La imagen f del ejemplo de transporte (skimage.data, sin conexión)."""
from skimage import color, data, transform


def imagen_f(L, imagen):
    """Imagen de skimage.data en grises, recortada al cuadrado central, L x L, en [0, 1]."""
    img = getattr(data, imagen)()
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    lado = min(img.shape)
    y0, x0 = (img.shape[0] - lado) // 2, (img.shape[1] - lado) // 2
    f = transform.resize(img[y0:y0 + lado, x0:x0 + lado].astype(float), (L, L), anti_aliasing=True)
    return (f - f.min()) / (f.max() - f.min())
