#!/usr/bin/env python3

import logging
import time

import funcy
import numpy as np
import pandas as pd

from horaha import PROJECT_ROOT
from semsch import Client

logger = logging.getLogger('horaha')

client = Client(
    cache=PROJECT_ROOT.joinpath('data', 'cache'))


def enable_logging(filename):
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    print('Logging to {} enabled.'.format(filename))


@funcy.memoize
def load_faculty_list():
    fn = PROJECT_ROOT.joinpath('data', 'faculty_list.txt')
    with fn.open('r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


@funcy.memoize
def load_author_metadata():
    fn = PROJECT_ROOT.joinpath('data', 'author_metadata.csv')
    dtype = {'name': str, 'authorId': int}
    df = pd.read_csv(fn, index_col=False, header=0, dtype=dtype)
    return df


@funcy.memoize
def author_name_to_id(name):
    author_metadata = load_author_metadata()
    return (author_metadata
            .set_index('name')
            .at[name, 'authorId'])


@funcy.memoize
def author_name_to_area(name):
    fn = PROJECT_ROOT.joinpath('data', 'author_areas.csv')
    dtype = {'name': str, 'area': str}
    df = pd.read_csv(fn, index_col=False, header=0, dtype=dtype)
    return (df
            .set_index('name')
            .at[name, 'area'])


@funcy.memoize
def get_coauthors(name, min_year=2008, max_year=2017):
    authorId = author_name_to_id(name)
    author = client.author(authorId)

    logger.info('Processing author {} - {}...'.format(authorId, name))
    coauthors = set()
    for paper in author.papers:
        if paper is None:
            continue

        try:
            year = int(paper.year)
        except (TypeError, ValueError):
            continue

        if not (min_year <= year and year <= max_year):
            continue

        logger.debug(
            'Downloading paper {id} - {title} ({year})...'
            .format(id=paper.paperId, title=paper.title,
                    year=paper.year))
        paper.load_data()
        for coauthor in paper.authors:
            coauthors.add(coauthor.authorId)

    return coauthors


def get_all_coauthors():
    faculty_list = load_faculty_list()
    result = {}
    for name in faculty_list:
        coauthors = get_coauthors(name)
        result[name] = coauthors
        time.sleep(3)

    return result


def get_coauthorship_matrix(coauthors=None):
    faculty_list = load_faculty_list()
    if coauthors is None:
        coauthors = get_all_coauthors()
    n = len(faculty_list)
    mat = np.zeros((n, n))
    for i in range(n):
        name_i = faculty_list[i]
        coauthors_i = coauthors[name_i]
        for j in range(i + 1, n):
            name_j = faculty_list[j]
            id_j = author_name_to_id(name_j)
            # i guess ids are extracted as ints...
            if str(id_j) in coauthors_i:
                mat[i, j] = 1
                mat[j, i] = 1

    return mat
