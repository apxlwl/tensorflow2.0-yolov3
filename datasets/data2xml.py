import cv2
import os
import numpy as np
from glob import iglob  # python 3.5 or newer
from shutil import copyfile
import matplotlib.pyplot as plt
# The script
import xml.etree.cElementTree as ET

def newXMLPASCALfile(imageheight, imagewidth, path, basename):
  annotation = ET.Element("annotation", verified="yes")
  ET.SubElement(annotation, "folder").text = "images"
  ET.SubElement(annotation, "filename").text = basename
  ET.SubElement(annotation, "path").text = path

  source = ET.SubElement(annotation, "source")
  ET.SubElement(source, "database").text = "test"

  size = ET.SubElement(annotation, "size")
  ET.SubElement(size, "width").text = str(imagewidth)
  ET.SubElement(size, "height").text = str(imageheight)
  ET.SubElement(size, "depth").text = "3"

  ET.SubElement(annotation, "segmented").text = "0"
  tree = ET.ElementTree(annotation)
  return tree


def appendXMLPASCAL(curr_et_object, x1, y1, w, h):
  et_object = ET.SubElement(curr_et_object.getroot(), "object")
  ET.SubElement(et_object, "name").text = "face"
  ET.SubElement(et_object, "pose").text = "Unspecified"
  ET.SubElement(et_object, "truncated").text = "0"
  ET.SubElement(et_object, "difficult").text = "0"
  bndbox = ET.SubElement(et_object, "bndbox")
  ET.SubElement(bndbox, "xmin").text = str(x1)
  ET.SubElement(bndbox, "ymin").text = str(y1)
  ET.SubElement(bndbox, "xmax").text = str(x1 + w)
  ET.SubElement(bndbox, "ymax").text = str(y1 + h)
  return curr_et_object


def readAndWrite(imgpath,annpath,bbx_gttxtPath):
  cnt = 0
  with open(bbx_gttxtPath, 'r') as f:
    curr_et_object = None
    annfile=None
    for line in f:
      inp = line.split(' ')
      if len(inp) == 1:
        if inp[0].find('jpg')==-1:
          continue
        if curr_et_object is not None and curr_et_object.find("object") is not None:
          curr_et_object.write(annfile)
        img_path = inp[0].strip('\n')
        filepath=imgpath + '/' + img_path
        img = cv2.imread(filepath, 2)  # POSIX only
        curr_filename = img_path.split("/")[1].strip()
        annfile = os.path.join(annpath, curr_filename.strip().replace(".jpg", ".xml"))
        dir_path = os.path.dirname(filepath)
        curr_et_object = newXMLPASCALfile(img.shape[0], img.shape[1], dir_path, curr_filename)
        del img
      else:
        inp = [int(i) for i in inp[:-1]]
        x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = inp
        n = max(w, h)
        if invalid == 1 or blur > 0 or n < 50:
          continue
        cnt += 1
        curr_et_object = appendXMLPASCAL(curr_et_object, x1, y1, w, h)


# ################################ TRAINING DATA 9263 ITEMS ##################################
# # # Run Script for Training data
subset=['train','val']
for set in subset:
  img_path = os.path.join("/home/gwl/datasets/WIDER", "WIDER_{}".format(set), "images")
  ann_path = os.path.join("/home/gwl/datasets/WIDER", "WIDER_{}".format(set), "annotations")
  if not os.path.exists(ann_path):
    os.mkdir(ann_path)
  ## comment this out
  bbx_gttxtPath = os.path.join("/home/gwl/datasets/WIDER", "wider_face_split", "wider_face_{}_bbx_gt.txt".format(set))
  readAndWrite(img_path,ann_path,bbx_gttxtPath)

