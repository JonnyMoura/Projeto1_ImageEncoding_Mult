import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.fftpack import dct, idct

matriz = np.array([[0.299, 0.587, 0.114],
                   [-0.168736, -0.331264, 0.5],
                   [0.5, -0.418688, -0.081312]])

Q_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q_CbCr = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def custom_colormap(colormap):
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colormap, N=256)

    return custom_cmap


def view_with_custom_cmap(img, cmap,title):
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.show()


def split_image(img):
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]
    return red_channel, green_channel, blue_channel


def combine_channels(red_channel, green_channel, blue_channel):
    return np.stack([red_channel, green_channel, blue_channel], axis=2)


def pad_image(img, multiple=32):
    original_shape = img.shape
    cols, rows = original_shape[0], original_shape[1]
    pad_rows = 0;
    pad_cols = 0;
    if (rows % multiple != 0):
        pad_rows = multiple - rows % multiple
    if (cols % multiple != 0):
        pad_cols = multiple - cols % multiple

    padded_img = np.pad(img, [(0, pad_cols), (0, pad_rows), (0, 0)], mode='edge')
    return padded_img, original_shape


def remove_padding(img, originalShape):
    if img.ndim == 3:
        img = img[:originalShape[0], :originalShape[1], :]
    else:
        img = img[:originalShape[0], :originalShape[1]]

    return img


def rgb_to_ycbcr(img):
    r, g, b = split_image(img)
    

    Y = matriz[0, 0] * r + matriz[0, 1] * g + matriz[0, 2] * b

    Cb = matriz[1, 0] * r + matriz[1, 1] * g + matriz[1, 2] * b + 128

    Cr = matriz[2, 0] * r + matriz[2, 1] * g + matriz[2, 2] * b + 128

    img_ycbcr = np.zeros((img.shape[0], img.shape[1], 3))
    img_ycbcr[:, :, 0] = Y
    img_ycbcr[:, :, 1] = Cb
    img_ycbcr[:, :, 2] = Cr

    return img_ycbcr


def ycbcr_to_rgb(img):
    matrizI = np.linalg.inv(matriz)

    Y, cb, cr = split_image(img)

    r = matrizI[0, 0] * Y + matrizI[0, 1] * (cb - 128) + matrizI[0, 2] * (cr - 128)
    g = matrizI[1, 0] * Y + matrizI[1, 1] * (cb - 128) + matrizI[1, 2] * (cr - 128)
    b = matrizI[2, 0] * Y + matrizI[2, 1] * (cb - 128) + matrizI[2, 2] * (cr - 128)

    img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)

    img_rgb[:, :, 0] = r
    img_rgb[:, :, 1] = g
    img_rgb[:, :, 2] = b

    img_rgb[img_rgb > 255] = 255
    img_rgb[img_rgb < 0] = 0
    return np.round(img_rgb).astype(np.uint8)


def downsampler(Y, Cb, Cr, yfactor, crfactor, cbfactor):
    if cbfactor == 0:  ##Downsample 4:2:0

        Y_d = Y
        cb_d = cv2.resize(Cb, None, fx=crfactor / yfactor, fy=crfactor / yfactor, interpolation=cv2.INTER_LINEAR)
        cr_d = cv2.resize(Cr, None, fx=crfactor / yfactor, fy=crfactor / yfactor, interpolation=cv2.INTER_LINEAR)
    else:  # Downsample 4:2:2
        Y_d = Y
        cb_d = cv2.resize(Cb, None, fx=cbfactor / yfactor, fy=1, interpolation=cv2.INTER_LINEAR)
        cr_d = cv2.resize(Cr, None, fx=crfactor / yfactor, fy=1, interpolation=cv2.INTER_LINEAR)

    return Y_d, cb_d, cr_d


def upsampler(Y_d, cb_d, cr_d, yfactor, crfactor, cbfactor):
    ##Upsample de 4:2:0
    if cbfactor == 0:

        Cb_u = cv2.resize(cb_d, None, fx=int(yfactor / crfactor), fy=int(yfactor / crfactor),
                          interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(cr_d, None, fx=int(yfactor / crfactor), fy=int(yfactor / crfactor),
                          interpolation=cv2.INTER_LINEAR)

        # 4:2:2
    else:
        Cb_u = cv2.resize(cb_d, None, fx=int(yfactor / cbfactor), fy=1,
                          interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(cr_d, None, fx=int(yfactor / crfactor), fy=1,
                          interpolation=cv2.INTER_LINEAR)

    return Y_d, Cb_u, Cr_u


def dct2(canal):
    return dct(dct(canal, norm='ortho').T, norm='ortho').T


def idct2(canal):
    return idct(idct(canal, norm='ortho').T, norm='ortho').T


def apply_dct_blocks(canal, BS):
    new_canal = np.zeros(canal.shape)

    h, w = canal.shape

    # make sure of the padding
    if h % BS != 0 or w % BS != 0:
        raise Exception("Canal de forma" + str(canal.shape) + "nao multiplo de " + str(BS))

    for i in range(0, h, BS):
        for j in range(0, w, BS):
            new_canal[i:i + BS, j:j + BS] = dct2(canal[i:i + BS, j:j + BS])

    return new_canal


def apply_idct_blocks(canal, BS):
    new_canal = np.zeros(canal.shape)

    h, w = canal.shape

    # make sure of the padding
    if h % BS != 0 or w % BS != 0:
        raise Exception("Canal de forma" + str(canal.shape) + "nao multiplo de " + str(BS))

    for i in range(0, h, BS):
        for j in range(0, w, BS):
            new_canal[i:i + BS, j:j + BS] = idct2(canal[i:i + BS, j:j + BS])

    return new_canal


def quality_factor(qf):
    if(qf==100):
        return np.ones((8,8)).astype(np.uint8),np.ones((8,8)).astype(np.uint8)
    if (qf >= 50):
        sf = (100 - qf) / 50
    else:
        sf = 50 / qf

    if (sf != 0):
        Q_Y_qf = np.round(Q_Y * sf)
        Q_CbCr_qf = np.round(Q_CbCr * sf)
    else:
        Q_Y[:, :] = 1
        Q_CbCr[:, :] = 1

    Q_Y_qf[Q_Y_qf > 255] = 255
    Q_Y_qf[Q_Y_qf < 1] = 1

    Q_CbCr_qf[Q_CbCr_qf > 255] = 255
    Q_CbCr_qf[Q_CbCr_qf < 1] = 1

    return Q_Y_qf.astype(np.uint8), Q_CbCr_qf.astype(np.uint8)


def quantization(channel, Q_qf, BS):
    new_canal = np.zeros(channel.shape)

    h, w = channel.shape

    # make sure of the padding
    if h % BS != 0 or w % BS != 0:
        raise Exception("Canal de forma" + str(channel.shape) + "nao multiplo de " + str(BS))

    for i in range(0, h, BS):
        for j in range(0, w, BS):
            new_canal[i:i + BS, j:j + BS] = np.round(channel[i:i + BS, j:j + BS] / Q_qf)

    return new_canal


def dequantization(channel, Q_qf, BS):
    new_canal = np.zeros(channel.shape).astype(np.float64)

    h, w = channel.shape

    # make sure of the padding
    if h % BS != 0 or w % BS != 0:
        raise Exception("Canal de forma" + str(channel.shape) + "nao multiplo de " + str(BS))

    for i in range(0, h, BS):
        for j in range(0, w, BS):
            new_canal[i:i + BS, j:j + BS] = channel[i:i + BS, j:j + BS] * Q_qf

    return new_canal

def dpcm(BS,channel):
    new_canal=channel.copy()
    h, w = channel.shape
    if h % BS != 0 or w % BS != 0:
        raise Exception("Canal de forma " + str(channel.shape) + " não é múltiplo de " + str(BS))

    prev_dc = 0
    for i in range(0, h, BS):
        for j in range(0, w, BS):
            new_canal[i:i + BS, j:j + BS][0,0] = channel[i:i + BS, j:j + BS][0,0]-prev_dc
            prev_dc= channel[i:i + BS, j:j + BS][0,0]


    return new_canal

def idpcm(BS,channel):
    new_canal = channel.copy()
    h, w = channel.shape
    if h % BS != 0 or w % BS != 0:
        raise Exception("Canal de forma " + str(channel.shape) + " não é múltiplo de " + str(BS))

    prev_dc = 0
    for i in range(0, h, BS):
        for j in range(0, w, BS):
            current_dc=channel[i:i + BS, j:j + BS][0, 0]
            new_canal[i:i + BS, j:j + BS][0, 0] =current_dc + prev_dc
            prev_dc = new_canal[i:i + BS, j:j + BS][0, 0]

    return new_canal

def encoder(img, multiple, factor_y, factor_cr, factor_cb, BS, qf):


    # padding
    paddedImage, originalShape = pad_image(copy.copy(img), multiple)

    # rgb -> YCbCr
    ycbcrImage = rgb_to_ycbcr(np.asarray(paddedImage))

    # Split channels
    Y, Cb, Cr = split_image(ycbcrImage)
   

    # downsamplig
    Y_d, Cb_d, Cr_d = downsampler(Y, Cb, Cr, factor_y, factor_cr, factor_cb)

    # apply dct
    if (BS == 0):
        Y_dct = dct2(Y_d)
        Cb_dct = dct2(Cb_d)
        Cr_dct = dct2(Cr_d)
    else:
        Y_dct = apply_dct_blocks(Y_d, BS)
        Cb_dct = apply_dct_blocks(Cb_d, BS)
        Cr_dct = apply_dct_blocks(Cr_d, BS)

    # Quantization
    Q_y, Q_cbcr = quality_factor(qf)
    Y_quanti = quantization(Y_dct, Q_y, BS)
    Cb_quanti = quantization(Cb_dct, Q_cbcr, BS)
    Cr_quanti = quantization(Cr_dct, Q_cbcr, BS)


    #DPCM
    Y_dpcm=dpcm(BS, Y_quanti)
    Cb_dpcm = dpcm(BS, Cb_quanti)
    Cr_dpcm = dpcm(BS, Cr_quanti)


    view_with_custom_cmap(Y_d, custom_colormap([(0, 0, 0), (1, 1, 1)]),"Y Channel-downsampled 4:2:2")
    view_with_custom_cmap(Cb_d, custom_colormap([(0, 0, 0), (1, 1, 1)]),"Cb Channel-downsampled 4:2:2")
    view_with_custom_cmap(Cr_d, custom_colormap([(0, 0, 0), (1, 1, 1)]),"Cr Channel-downsampled 4:2:2")

    return originalShape, Y_dpcm, Cb_dpcm, Cr_dpcm, factor_y, factor_cr, factor_cb, BS, qf


def decoder(originalShape, Y, Cb, Cr, factor_y, factor_cr, factor_cb, BS, qf, *, show_img=False):
    #reverse dpcm
    Y_quanti=idpcm(BS,Y)
    Cb_quanti=idpcm(BS,Cb)
    Cr_quanti=idpcm(BS,Cr)
    # reverse quantization
    Q_y, Q_cbcr = quality_factor(qf)
    Y_dct = dequantization(Y_quanti, Q_y, BS)
    Cb_dct = dequantization(Cb_quanti, Q_cbcr, BS)
    Cr_dct = dequantization(Cr_quanti, Q_cbcr, BS)

    # inverse DCT
    if (BS == 0):
        Y_idct = idct2(Y_dct)
        Cb_idct = idct2(Cb_dct)
        Cr_idct = idct2(Cr_dct)

    else:
        Y_idct = apply_idct_blocks(Y_dct, BS)
        Cb_idct = apply_idct_blocks(Cb_dct, BS)
        Cr_idct = apply_idct_blocks(Cr_dct, BS)

    # upsampling
    Y_idct_u, Cb_idct_u, Cr_idct_u = upsampler(Y_idct, Cb_idct, Cr_idct, factor_y, factor_cr, factor_cb)

    # remove padding
    Y_r = remove_padding(Y_idct_u, originalShape)
    Cb_r = remove_padding(Cb_idct_u, originalShape)
    Cr_r = remove_padding(Cr_idct_u, originalShape)

    #ycbcr to rgb
    rgb_r=ycbcr_to_rgb(combine_channels(Y_r,Cb_r,Cr_r))
    
    if show_img:
        plt.imshow(rgb_r)
        plt.show()

    return rgb_r
    pass

def show_error(img: np.ndarray, img_rec, *, show_error_image=False):
    Y, _, _ = split_image(rgb_to_ycbcr(np.asarray(img)))
    Y_rec, _, _ = split_image(rgb_to_ycbcr(np.asarray(img_rec)))
    erro_ma = np.abs(Y - Y_rec)
    
    img = img.astype(np.float32)

    a, b, _ = img.shape
    MSE = np.sum((img - img_rec)**2)/(a*b)
    RMSE = MSE**.5
    
    print("MSE:", MSE)
    print("RMSE:", RMSE)

    P = np.sum(img**2)/(a*b)
    SNR = 10*np.log10(P/MSE)
    print("SNR:", SNR)

    PSNR = 10*np.log10(img.max()**2/MSE)
    print("PSNR:", PSNR)

    if show_error_image:
        plt.imshow(erro_ma, cmap=custom_colormap([(0, 0, 0), (1, 1, 1)]))
        plt.title(str(show_error_image))
        plt.show()



def main():
    img = plt.imread('./imagens/barn_mountains.bmp')
    # img = plt.imread('./imagens/logo.bmp')
    # img = plt.imread('./imagens/peppers.bmp')
    # img = plt.imread('./imagens/peppers.jpg')

    
    for i in [10, 25, 50, 75, 100]:
        *info, = encoder(img, 32, 4, 2, 2, 8, i)
        img_rec = decoder(*info)
        
        print(f"{i}:")
        show_error(img, img_rec, show_error_image=i)

        print("-"*14)


if __name__ == '__main__':
    main()