import unittest

from semsch import Client

class TestClient(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_author(self):
        authorId = '35651709'
        author = self.client.author(authorId)

        self.assertEqual(author.authorId, authorId)
        self.assertEqual(author.name, 'Micah J. Smith')
        self.assertEqual(
            author.url,
            'https://www.semanticscholar.org/author/{}'.format(authorId))

    def test_paper(self):
        paperId = '62e62f92df3e5d46346a5d4c2d7a8be3a50e1cac'
        paper = self.client.paper(paperId)

        self.assertEqual(paper.paperId, paperId)
        self.assertEqual(
            paper.title, 'FeatureHub: Towards Collaborative Data Science')
        self.assertEqual(
            paper.url,
            'https://www.semanticscholar.org/paper/{}'.format(paperId))
