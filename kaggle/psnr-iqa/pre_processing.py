# Effective_read_images
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('sample_submission.csv')
print('train.shape test.shape', train.shape, test.shape)

def brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].mean()
    return v / 255 # normalization

def sharpness_grad_based(img):
    gradx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    grady = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    grad = np.sqrt(gradx * gradx + grady * grady)
    return grad[1: -1, 1: -1].mean()

def svd(img, num):
    channel = 3

    singular_values = []

    for i in range(channel):
        U, s, VT = np.linalg.svd(img[:, :, i])

        # k = 10
        # S = np.diag(s[:k])
        # B = np.dot(U[:, :k], np.dot(S, VT[:k, :]))

        # cv2.imshow('img', img[:, :, i])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow('B', B.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # print('s[:k]', s[:k])
        # print('sum', s[:k].sum())
        # print('mean', s[:k].mean())
        # print('1', s[0] / s[:k].sum())
        #
        # print('sum', s.sum())
        # print('2', s[0] / s.sum())

        # co_coef = 0.
        # for j in range(20):
        #     co = s[j] / s.sum()
        #     co_coef += co
        #
        # co_coef = co_coef / 20


        # co_coef = s[0] / s.sum()
        # print(i, co_coef)

        singular_values.append(s)

    # print()
    # print('singular_values', singular_values)

    # return singular_values[0], singular_values[1], singular_values[2]

    # print('singular_values')
    # print(singular_values)

    # # 합
    # c = [(x1 + x2 + x3) / channel for x1, x2, x3 in zip(singular_values[0], singular_values[1], singular_values[2])]

    # 따로
    c1 = singular_values[0]
    c2 = singular_values[1]
    c3 = singular_values[2]

    return c1[:num], c2[:num], c3[:num]


base_path = 'SBC22_IQA_dataset'

count = 0

def get_columns_from_image(image: pd.Series, folder):
    full_path = os.path.join(base_path, folder, image)
    img = cv2.imread(full_path)

    b = brightness(img)
    s = sharpness_grad_based(img)
    # sh_1, sh_2, sh_3 = svd(img)
    # print('sh_1, sh_2, sh_3', sh_1, sh_2, sh_3)

    # sh = svd(img, 10)
    # sh = svd(img, img.shape[0])
    ch1_sh, ch2_sh, ch3_sh = svd(img, img.shape[0])

    # for cnt, i in zip(range(len(sh)), sh):
    #     print(cnt, i)
    # plt.figure()
    # plt.plot(np.arange(len(sh)), sh)
    # plt.title('eigenvalues')
    # plt.show()
    # print(img.shape)

    global count
    print('count', count)
    count += 1

    results_df = pd.Series({
        'brightness': b,
        'sharpness_grad_based': s
    })

    # for cnt in range(len(sh)):
    #     results_df['sh_' + str(cnt)] = sh[cnt]

    for cnt in range(len(ch1_sh)):
        results_df['ch1_sh_' + str(cnt)] = ch1_sh[cnt]
    for cnt in range(len(ch2_sh)):
        results_df['ch2_sh_' + str(cnt)] = ch2_sh[cnt]
    for cnt in range(len(ch3_sh)):
        results_df['ch3_sh_' + str(cnt)] = ch3_sh[cnt]

    # return pd.Series({
    #     'brightness': b,
    #     'sharpness_grad_based': s,
    #     'sh_1': sh[0],
    #     'sh_2': sh[1],
    #     'sh_3': sh[2],
    #     'sh_4': sh[3],
    #     'sh_5': sh[4],
    #     'sh_6': sh[5],
    #     'sh_7': sh[6],
    #     'sh_8': sh[7],
    #     'sh_9': sh[8],
    #     'sh_10': sh[9]
    # })

    return results_df


count = 0
train = train.join(train['img_name'].apply(lambda x: get_columns_from_image(x, 'train')))
print('train.shape', train.shape)

train.to_csv('train_full.csv',index=False)

count = 0
test = test.join(test['img_name'].apply(lambda x: get_columns_from_image(x, 'test')))
print('test.shape', test.shape)
test.to_csv('test_full.csv',index=False)
