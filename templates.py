#!/usr/bin/env python


def image(id, filename, width=400, height=400):
    return dict(
        id=id,
        file_name=filename,
        width=width,
        height=height
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
    return { 'contributor': 'crowdAI.org',
             'about': 'Dataset derived from the spacenet v2.0 (http://explore.digitalglobe.com/spacenet)',
             'date_created': '07/03/2018',
             'description': 'crowdAI mapping-challenge dataset',
             'url': 'https://www.crowdai.org/challenges/mapping-challenge',
             'version': '1.0',
             'year': 2018}

def categories():
    return [{'id': 100, 'name': 'building', 'supercategory': 'building'}]
