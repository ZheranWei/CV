import onnxruntime
import cv2
import numpy as np
from torchvision import transforms
import torch

img_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((320, 480))])


def cpu_demo():
    device_name = onnxruntime.get_device()
    print(device_name)
    session = onnxruntime.InferenceSession(r"E:\PycharmProjects\pytorch\crack-segmentation\weights\road.onnx")
    image = cv2.imread("dataset/test/301.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    img = np.float32(image) / 255.0
    x_input = img_transform(img).view(1, 3, h, w)
    input = {session.get_inputs()[0].name: x_input.numpy()}
    output = session.run(None, input)
    m_label_out_ = torch.from_numpy(output[0]).transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
    _, predict = m_label_out_.data.max(dim=1)
    predict[predict > 0] = 255
    predic_ = predict.view(h, w).cpu().detach().numpy()

    cv2.imshow("input", image)
    result = cv2.resize(np.uint8(predic_), (w, h))
    h = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = h[1]
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr_img, contours, -1, (0, 0, 255), -1)
    cv2.imshow("output", bgr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output']):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct, please give a name that ends with ".onnx"!')
        return 0
    model = torch.load(checkpoint).cpu()
    model.eval()

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names, opset_version=11)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    checkpoint = 'weights/road.pt'
    onnx_path = 'weights/road1.onnx'
    input = torch.randn(1, 3, 320, 480)
    pth_to_onnx(input, checkpoint, onnx_path)
    cpu_demo()
