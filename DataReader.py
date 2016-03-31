import os
import numpy as np


class Data:
    def __init__(self):
        self._train = {}
        self._valid = {}
        self._test = {}
        self._max_id = 0
        self._id2name = {}
        self._name2id = {}

    def load_data(self, folder="./data/FB15k"):
        raise NotImplementedError("Abstract method")

    @property
    def max_id(self):
        """max id of items in data"""
        return self._max_id

    @property
    def train(self):
        """training dataset"""
        return self._train

    @property
    def test(self):
        """testing dataset"""
        return self._test

    @property
    def valid(self):
        """validation dataset"""
        return self._valid

    @property
    def name2id(self):
        """map a string to an id"""
        return self._name2id

    @property
    def id2name(self):
        """map an id to a string"""
        return self._id2name

    @staticmethod
    def _id_loader(path, init_storage=None, init_rstorage=None):
        """Load a list of elements and assign 0-indexed id to them.
        NOTE: ID-0 IS THE PLACEHOLDER INDEX FOR NULL
        :param path: File path, each line contains an element
        :return: A element->id dict and a id->element dict
        """
        storage = dict()
        rstorage = dict()
        sid = 0

        if init_storage is not None and init_rstorage is not None:
            assert len(init_storage) == len(init_rstorage)

            storage = init_storage
            rstorage = init_rstorage
            sid = max(storage.values()) + 1

        with open(path) as f:
            for line in f:
                s = line.strip()
                storage[s] = sid
                rstorage[sid] = s
                sid += 1
        return storage, rstorage

    @staticmethod
    def _path_loader(path, entities, relations=None):
        """Load paths with structure `head tail predicate`.
        :param path: File path
        :return: A list of lists, each list is a length-3 tuple representing a path.
                 An adjacent list contains the relations/predicates that connects head and tail
        """
        storage = []
        with open(path) as f:
            for line in f:
                head, tail, rel = line.rstrip().split()
                head = entities[head]
                tail = entities[tail]
                rel = relations[rel] if relations is not None else entities[rel]
                storage.append([head, tail, rel])

        return storage

    @staticmethod
    def gen_adj_list(dat):

        adj_list = dict()

        for edge in dat:
            head, tail, rel = edge
            if head not in adj_list:
                adj_list[head] = dict()
            if tail not in adj_list[head]:
                adj_list[head][tail] = []
            adj_list[head][tail].append(rel)

        return adj_list

    @staticmethod
    def gen_lmap(dat):

        hl_map = dict()  # [head][rel] => tails
        tl_map = dict()  # [tail][rel] => heads

        for edge in dat:
            head, tail, rel = edge

            # hl_map
            if head not in hl_map:
                hl_map[head] = dict()
            if rel not in hl_map[head]:
                hl_map[head][rel] = set()

            hl_map[head][rel].add(tail)

            # tl_map
            if tail not in tl_map:
                tl_map[tail] = dict()
            if rel not in tl_map[tail]:
                tl_map[tail][rel] = set()

            tl_map[tail][rel].add(head)

        return hl_map, tl_map


class MetaPathData(Data):
    _hlmap = dict()
    _tlmap = dict()

    _hl_test_map = dict()
    _tl_test_map = dict()

    _entity2id = dict()
    _id2entity = dict()

    _entity_id_min = 0
    _entity_id_max = 0

    _rel2id = dict()
    _id2rel = dict()

    _rel_id_min = 0
    _rel_id_max = 0

    inputs = None
    targets = None
    corrupted = None
    corrupted_target = None

    h = None
    t = None
    h_prime = None
    t_prime = None
    l = None

    __rnn_data = None

    def load_data(self, folder="./data/FB15k"):
        entity_path = os.path.join(folder, "entity.txt")
        relation_path = os.path.join(folder, "relation.txt")
        train_path = os.path.join(folder, "train.txt")
        valid_path = os.path.join(folder, "valid.txt")
        test_path = os.path.join(folder, "test.txt")

        self._entity_id_min = 0
        self._entity2id, self._id2entity = Data._id_loader(entity_path)  # ignored
        self._entity_id_max = len(self.entity2id) - 1

        self._rel_id_min = 0
        self._rel2id, self._id2rel = Data._id_loader(relation_path)
        self._rel_id_max = len(self.rel2id) - 1

        train = Data._path_loader(train_path, self.entity2id, self.rel2id)
        train_adj = Data.gen_adj_list(train)

        test = Data._path_loader(test_path, self.entity2id, self.rel2id)
        test_adj = Data.gen_adj_list(test)

        valid = Data._path_loader(valid_path, self.entity2id, self.rel2id)
        valid_adj = Data.gen_adj_list(valid)

        self._train = {'path': np.asarray(train), 'adj': train_adj}
        self._valid = {'path': np.asarray(valid), 'adj': valid_adj}
        self._test = {'path': np.asarray(test), 'adj': test_adj}

        #  make sure if someone calls these variables it will cause an error
        self._max_id = None
        self._name2id = None
        self._id2name = None

        self._hlmap, self._tlmap = Data.gen_lmap(self._train['path'])
        self._hl_test_map, self._tl_test_map = Data.gen_lmap(self._test['path'])
        self.hl_valid_map, self.tl_valid_map = Data.gen_lmap(self._valid['path'])

    @property
    def entity2id(self):
        return self._entity2id

    @property
    def id2entity(self):
        return self._id2entity

    @property
    def rel2id(self):
        return self._rel2id

    @property
    def id2rel(self):
        return self._id2rel

    @property
    def max_id(self):
        raise DeprecationWarning("Please use entity_id_max or rel_id_max instead.")

    @property
    def id2name(self):
        raise DeprecationWarning("Please use id2entity or id2rel instead.")

    @property
    def name2id(self):
        raise DeprecationWarning("Please use entity2id or rel2id instead.")

    @property
    def hlmap(self):
        """header,relation => tails"""
        return self._hlmap

    @property
    def tlmap(self):
        """tail,relation => heads"""
        return self._tlmap

    @property
    def hl_test_map(self):
        return self._hl_test_map

    @property
    def tl_test_map(self):
        return self.tl_test_map

    @property
    def entity_id_min(self):
        return self._entity_id_min

    @property
    def entity_id_max(self):
        return self._entity_id_max

    @property
    def rel_id_min(self):
        return self._rel_id_min

    @property
    def rel_id_max(self):
        return self._rel_id_max
