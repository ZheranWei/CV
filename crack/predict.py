import cv2 as cv
import os
import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import cvtColor


def unet_defect_demo():
    model = torch.load("weights/road.pt")
    test_dir = "dataset/test"
    fileNames = os.listdir(test_dir)
    for f in fileNames:
        image = cvtColor(cv.imread(os.path.join(test_dir, f)))
        h, w, _ = image.shape
        img = np.float32(image) / 255.0
        img = np.transpose(img, [2, 0, 1])

        x_input = torch.from_numpy(img).view(1, 3, h, w)
        probs = model(x_input.cuda())
        c, h, w = img.shape
        nt, ct, ht, wt = probs.size()
        if h != ht and w != wt:
            probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=True)
        m_label_out_ = probs.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)

        _, output = m_label_out_.data.max(dim=1)
        output[output > 0] = 255
        predic_ = output.view(h, w).cpu().detach().numpy()

        cv.imshow("input", image)
        result = cv.resize(np.uint8(predic_), (w, h))
        h = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = h[1]
        bgr_img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.drawContours(bgr_img, contours, -1, (0, 0, 255), -1)
        cv.imshow("output", bgr_img)
        cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    unet_defect_demo()
