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
        is_crowd=is_crowd
    )
