from xml.etree import ElementTree as ET
import cv2
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms


def parse_xml(xml_fp):
    bnd_boxes = []
    for node in ET.parse(xml_fp).getroot():
        if node.tag == "object":
            bnd_boxes.append([int(item.text) for item in node[4]])

    return bnd_boxes


def load_img_and_xml(img_list, xml_list):
    imgs, xmls = [], []
    for img_fp, xml_fp in zip(img_list, xml_list):
        img = transforms.ToTensor()(Image.open(img_fp))
        xml = None if xml_fp is None else parse_xml(xml_fp)
        imgs.append(img)
        xmls.append(xml)
    return imgs, xmls


def check_bbox_intersect(bA, bB):
    return not (bA[0] > bB[2] or bA[2] < bB[0] or bA[1] > bB[3] or bA[3] < bB[1])


def trim_box(bbox, width, height):
    if bbox[0] < 0:
        dx = bbox[0] - 0
    elif bbox[2] > width - 1:
        dx = bbox[2] - width + 1
    else:
        dx = 0
    if bbox[1] < 0:
        dy = bbox[1] - 0
    elif bbox[3] > height - 1:
        dy = bbox[3] - height + 1
    else:
        dy = 0
    bbox[0] -= dx
    bbox[2] -= dx
    bbox[1] -= dy
    bbox[3] -= dy


class XLDataset():
    def __init__(self, imgs, xmls, df_ratio=0.5, psize=224):
        self.imgs = imgs
        self.xmls = xmls
        self.df_ratio = df_ratio
        self.psize = psize
        self.length = len(self.xmls)

    def __len__(self):
        # return 2 ** 32 - 1
        return 320000

    def __getitem__(self, index):
        ii = index % self.length
        img = self.imgs[ii]
        height, width = img.size(1), img.size(2)
        # print(img.size())
        bbox = random.sample(self.xmls[ii], 1)[0]
        df_flag = np.random.uniform(0.0, 1.0) < self.df_ratio

        bx0, by0, bx1, by1 = bbox

        if df_flag:
            cx = random.sample(range(bx0, bx1), 1)[0]
            cy = random.sample(range(by0, by1), 1)[0]
            x0 = cx - self.psize // 2
            x1 = cx + self.psize // 2
            y0 = cy - self.psize // 2
            y1 = cy + self.psize // 2
            random_bbox = [x0, y0, x1, y1]
            # print(random_bbox)
            trim_box(random_bbox, width, height)
            # print(random_bbox)
            # patch = img[:, random_bbox[1]:random_bbox[3], random_bbox[0]:random_bbox[2]]
        else:
            while True:
                cx = random.sample(range(0, width - self.psize), 1)[0]
                cy = random.sample(range(0, height - self.psize), 1)[0]
                x0 = cx - self.psize // 2
                x1 = cx + self.psize // 2
                y0 = cy - self.psize // 2
                y1 = cy + self.psize // 2
                random_bbox = [x0, y0, x1, y1]
                trim_box(random_bbox, width, height)
                if check_bbox_intersect(random_bbox, bbox):
                    continue
                else:
                    break
        patch = img[:, random_bbox[1]:random_bbox[3], random_bbox[0]:random_bbox[2]]
        # print(random_bbox[2] - random_bbox[0], random_bbox[3] - random_bbox[1])
        # print(random_bbox)
        return patch, 1 if df_flag else 0


if __name__ == '__main__':
    imgs = [transforms.ToTensor()(Image.open("../data/sample/J01_2018.06.27 15_20_25.jpg"))]
    xmls = [parse_xml("../data/sample/J01_2018.06.27 15_20_25.xml")]
    ds = XLDataset(imgs, xmls)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    for item in dl:
        # print(item.size())
        print("ahha")
        cv2.imshow("hehe", item[0].mean(0).numpy())
        cv2.waitKey()
