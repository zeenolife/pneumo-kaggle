import numpy as np
import pandas as pd
import json
import pydicom
import os
import cv2
import uuid
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa


NUM_AUG = 4
NUM_FOLDS = 10


augmentation = iaa.Sequential([
    iaa.OneOf([
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-5, 5),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])


class COCODataset:
    def __init__(self):

        self._classes = {}
        self._images = {}

        self._train_set = {'info': None,
                           'categories': [],
                           'images': [],
                           'annotations': [],
                           }
        self._test_set = {'info': None,
                          'categories': [],
                          'images': [],
                          'annotations': [],
                          }

        self._photo_idx = 201800000
        self._category_idx = 1
        self._annotation_idx = 1000000000

    def _add_image(self, img_name, img, to_train):

        h, w = img.shape[:2]

        if to_train:
            self._train_set['images'].append({u'file_name': img_name,
                                              u'height': h,
                                              u'width': w,
                                              u'id': self._photo_idx,
                                              })
        else:
            self._test_set['images'].append({u'file_name': img_name,
                                             u'height': h,
                                             u'width': w,
                                             u'id': self._photo_idx,
                                             })

        self._images[img_name] = self._photo_idx
        self._photo_idx += 1

    def _add_category(self, cls_name):

        if cls_name not in self._classes:
            self._classes[cls_name] = self._category_idx
            self._category_idx += 1

            self._train_set['categories'].append({u'id': self._classes[cls_name],
                                                  u'name': cls_name,
                                                  u'supercategory': u'none', })
            self._test_set['categories'].append({u'id': self._classes[cls_name],
                                                 u'name': cls_name,
                                                 u'supercategory': u'none', })

    def _add_annotation(self, coor, cls_name, img_name, to_train=True):

        x1, y1, x2, y2 = coor
        w = x2 - x1
        h = y2 - y1

        area = w * h
        bbox = [x1, y1, w, h]

        category_id = self._classes[cls_name]

        annot_id = self._annotation_idx
        self._annotation_idx += 1

        image_id = self._images[img_name]

        is_crowd = 0

        if to_train:
            self._train_set['annotations'].append({u'area': area,
                                                   u'bbox': bbox,
                                                   u'category_id': category_id,
                                                   u'id': annot_id,
                                                   u'image_id': image_id,
                                                   u'iscrowd': is_crowd,
                                                   })

        else:
            self._test_set['annotations'].append({u'area': area,
                                                  u'bbox': bbox,
                                                  u'category_id': category_id,
                                                  u'id': annot_id,
                                                  u'image_id': image_id,
                                                  u'iscrowd': is_crowd,
                                                  })

    def add_instance(self, img_name, img, coors, cls_names, to_train=True):

        # Add image info
        self._add_image(img_name, img, to_train)

        # Iterate through objects
        for coor, cls_name, in zip(coors, cls_names):
            self._add_category(cls_name)
            self._add_annotation(coor, cls_name, img_name, to_train=to_train)

    def dump_sets(self, save_path, split_idx):

        if len(self._train_set['images']) < 1:
            print('Train set is empty. Not dumping')
        else:
            print('Dumping train detection set')
            with open(os.path.join(save_path, 'train' + str(split_idx) + '.json'), 'w') as f:
                json.dump(self._train_set, f)

        if len(self._test_set['images']) < 1:
            print('Test set is empty. Not dumping')
        else:
            print('Dumping test set')
            with open(os.path.join(save_path, 'test' + str(split_idx) + '.json'), 'w') as f:
                json.dump(self._test_set, f)

        print 'Done'


def main(df, path_imgs, path_save):
    
    # Misc pre-augmentation stuff
    total_imgs = df['patientId'].nunique()
    
    # 10 fold split. Make it deterministic
    np.random.seed(63)
    perm = np.random.permutation(total_imgs)
    assert total_imgs > NUM_FOLDS, 'Number of folds cannot be more than images'
    fold_step = int(total_imgs / NUM_FOLDS)
    
    for split_idx in tqdm(range(NUM_FOLDS)):
        
        left = split_idx * fold_step
        right = left + fold_step
        
        val_set_ids = set(perm[left:right])

        if not os.path.isdir(os.path.join(path_save, 'train' + str(split_idx))):
            os.mkdir(os.path.join(path_save, 'train' + str(split_idx)))
    
        if not os.path.isdir(os.path.join(path_save, 'test' + str(split_idx))):
            os.mkdir(os.path.join(path_save, 'test' + str(split_idx)))

        # Main info
        dataset = COCODataset()

        # Iterate all photos
        for patient_idx, patient in enumerate(df.groupby('patientId')):
    
            patient_id, patient_df = patient
    
            # Read image and check train/test
            img_path = os.path.join(path_imgs, patient_id + '.dcm')
            img = cv2.cvtColor(pydicom.read_file(img_path).pixel_array, cv2.COLOR_GRAY2RGB)

            assert img is not None, 'Image is None: {}'.format(img_path)
    
            to_train = False if patient_idx in val_set_ids else True

            # Iterate the objects
            coors = []
            clss = []
            for index, row in patient_df.iterrows():

                x1 = int(float(row['x']))
                y1 = int(float(row['y']))
                x2 = int(float(row['width'])) + x1
                y2 = int(float(row['height'])) + y1
                clss.append('pneumonia')

                coors.append([x1, y1, x2, y2])

            # Save the original image
            img_name = str(uuid.uuid4()) + '.jpg'
            if to_train:
                cv2.imwrite(os.path.join(path_save, 'train' + str(split_idx), img_name), img)
                dataset.add_instance(img_name, img, coors, clss, to_train=to_train)
            else:
                cv2.imwrite(os.path.join(path_save, 'test' + str(split_idx), img_name), img)
                dataset.add_instance(img_name, img, coors, clss, to_train=to_train)
    
            # Skip the test images
            if not to_train:
                continue
    
            aug_det = augmentation.to_deterministic()

            bbox_list = [ia.BoundingBox(x1=_[0], y1=_[1], x2=_[2], y2=_[3]) for _ in coors]
            bbs = ia.BoundingBoxesOnImage(bbox_list, shape=img.shape)

            img_augs = aug_det.augment_images([img] * NUM_AUG)
            bbs_augs = aug_det.augment_bounding_boxes([bbs] * NUM_AUG)

            # Save augmented images
            for img_aug, bbs_aug in zip(img_augs, bbs_augs):

                img_name = str(uuid.uuid4()) + '.jpg'
                cv2.imwrite(os.path.join(path_save, 'train' + str(split_idx), img_name), img_aug)
                aug_coors = []
                clss = []

                for bb in bbs_aug.bounding_boxes:
                    aug_coors.append([int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)])
                    clss.append('pneumonia')

                dataset.add_instance(img_name, img_aug, aug_coors, clss, to_train=True)

        dataset.dump_sets(path_save, split_idx)
