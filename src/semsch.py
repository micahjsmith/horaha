import json as _json
import pathlib

import funcy
import requests


__all__ = ['Client']


SEMSCH_API_BASE_URL = 'https://api.semanticscholar.org/v1/'


class Client:
    def __init__(self, cache=None):
        if cache is not None:
            self._cache= pathlib.Path(cache)
            self._get = self._get_api_or_get_cache
        else:
            self._get = self._get_api

    def author(self, authorId):
        return Author(authorId, self)

    def paper(self, paperId):
        return Paper(paperId, self)

    def _get_api(self, path):
        url = SEMSCH_API_BASE_URL + str(path)
        params = {}
        res = requests.get(url, params=params)
        res.raise_for_status()
        json = res.json()
        return json

    def _get_api_or_get_cache(self, path):
        abspath = self._cache.joinpath(path)
        if abspath.exists():
            with abspath.open('r') as f:
                return _json.load(f)
        else:
            json = self._get_api(path)
            self._put(path, json)
            return json

    def _put(self, path, json):
        abspath = self._cache.joinpath(path)
        abspath.parent.mkdir(parents=True, exist_ok=True)
        with abspath.open('w') as f:
            _json.dump(json, f)


class LazyAPIData(object):
    def __init__(self, lazy_attrs):
        self.lazy_attrs = set(lazy_attrs)
        self.json = None
        self.data = None

    def __getattr__(self, key):
        if key in self.lazy_attrs:
            if self.data is None:
                self.load_data(depth=1)
            return self.data[key]
        raise AttributeError(key)

    def load_data(self, depth=1):
        raise NotImplementedError

    def __repr__(self):
        return '<{module}.{klass} object> with data:\n{data}' .format(
            module=__name__, klass=self.__class__.__name__,
            data=repr(self.data))


class Author(LazyAPIData):
    lazy_attrs = [
        # 'aliases',
        'citationVelocity',
        'influentialCitationCount',
        'name',
        'papers',
        'url',
    ]

    def __init__(self, authorId, client):
        self.authorId = authorId
        self._client = client
        super().__init__(Author.lazy_attrs)

    def load_data(self, depth=1):
        if depth == 0:
            return

        self.json = self._client._get('author/{}'.format(self.authorId))
        data = {
            'papers': [
                Paper(id, self._client)
                for id in funcy.pluck('paperId', self.json['papers'])
                if id is not None
            ],
            'citationVelocity': self.json['citationVelocity'],
            'influentialCitationCount': self.json['influentialCitationCount'],
            'name': self.json['name'],
            'url': self.json['url'],
        }
        self.data = data

        if depth - 1 > 0:
            for paper in self.data['papers']:
                paper.load_data(depth=depth - 1)


class Paper(LazyAPIData):
    lazy_attrs = [
        'authors',
        'citationVelocity',
        # 'citations',
        'doi',
        'influentialCitationCount',
        # 'references',
        'title',
        'url',
        'venue',
        'year',
    ]

    def __init__(self, paperId, client):
        self.paperId = paperId
        self._client = client
        super().__init__(Paper.lazy_attrs)

    def load_data(self, depth=1):
        if depth == 0:
            return

        self.json = self._client._get('paper/{}'.format(self.paperId))
        data = {
            'authors': [
                Author(id, self._client)
                for id in funcy.pluck('authorId', self.json['authors'])
                if id is not None
            ],
            'citationVelocity': self.json['citationVelocity'],
            'doi': self.json['doi'],
            'influentialCitationCount': self.json['influentialCitationCount'],
            'title': self.json['title'],
            'url': self.json['url'],
            'venue': self.json['venue'],
            'year': self.json['year'],
        }
        self.data = data

        if depth - 1 > 0:
            for author in self.data['authors']:
                author.load_data(depth=depth - 1)
