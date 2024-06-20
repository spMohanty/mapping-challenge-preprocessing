#!/usr/bin/env python


def image(id, filename, width=300, height=300, augmentation_source_id=None):
    return dict(
        id=id,
        file_name=filename,
        width=width,
        height=height,
        augmentation_source_id=augmentation_source_id
    )

def annotataion(id, image_id, segmentation, area, bbox, \
                category_id=100, is_crowd=0):

    return dict(
        id=id,
        image_id=image_id,
        segmentation=segmentation,
        area=area,
        bbox=bbox,
        category_id=category_id,
        iscrowd=is_crowd
    )

def info():
    return { 'contributor': 'AIcrowd.com',
             'about': 'Corrected Dataset for crowdAI Mapping Challenge',
             'date_created': '20/06/2024',
             'description': 'crowdAI mapping-challenge dataset',
             'url': 'https://www.aicrowd.com/challenges/mapping-challenge',
             'version': '2.0',
             'year': 2024}

def categories():
    return [{'id': 100, 'name': 'building', 'supercategory': 'building'}]
