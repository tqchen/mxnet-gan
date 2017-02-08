"""Visualization modules"""

import cv2

import numpy as np

def _fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i//m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img


def layout(X, flip=False):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip(X * 255.0, 0, 255).astype(np.uint8)
    n = int(np.ceil(np.sqrt(X.shape[0])))
    buff = np.zeros((n*X.shape[1], n*X.shape[2], X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        img = np.flipud(img) if flip else img
        _fill_buf(buff, i, img, X.shape[1:3])
    if buff.shape[-1] == 1:
        return buff.reshape(buff.shape[0], buff.shape[1])
    if X.shape[-1] != 1:
        buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    return buff


def imshow(title, X, waitsec=1, flip=False):
    """Show images in X and wait for wait sec.
    """
    buff = layout(X, flip=flip)
    cv2.imshow(title, buff)
    cv2.waitKey(waitsec)
