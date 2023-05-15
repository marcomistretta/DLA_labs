import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms.functional as TF

# same 10 classes of cifar
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
cifar = True  # xxx else i have an high quality truck image
if cifar:
    # 1 ship
    # 7 frog
    # 27 airplane
    # 42 dog
    image_idx = 18  # indice dell'immagine da testare
    transform = transforms.Compose(
        [transforms.ToTensor()])
    test_set = CIFAR10(root='./data', train=False,
                       download=True, transform=transform)
    image, label = test_set[image_idx]
    pil_image = TF.to_pil_image(image)
    # Display the image
    plt.imshow(pil_image)
    plt.show()
    # Save the image
    image_file = 'images/cifar_' + str(classes[label]) + '.jpg'
    pil_image.save(image_file)
    print("real label:", classes[label])
else:
    image_file = 'images/hd_truck.jpg'
    print("using hd image:", image_file)

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.


finalconv_name = "features"
# net = torch.load("./model/resnet_cnn-ep5-lr0.001-bs512-depth5-residual.pt")
net = torch.load("./model/resnet_to_convergence/cnn-ep5-lr0.004-bs64-depth25-residual.pt")
print(net)
net.eval()

# hook the feature extractor
features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
# )
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # normalize
])

# load test image
img_pil = Image.open(image_file)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable.to('cuda'))

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()

# output the prediction
for i in range(0, 10):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
img = cv2.imread(image_file)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.3 + img * 0.5
result = heatmap * 0.4 + img * 0.5
if cifar:
    cv2.imwrite('images/CAM_cifar_' + str(classes[label]) + '_idx' + str(image_idx) + '_probs' + str(probs[0]) + '.jpg',
                result)
else:
    cv2.imwrite('images/CAM_hd_truck_probs' + str(probs[0]) + '.jpg', result)
