from project_settings import HParams, DatasetConfig
from utils import copy_file,save_file,load_file,list_all_files
import re
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils import parse_xml
import pdb
from project_settings import EOS_TOK,EOC_TOK




class AUSDataset():
    def __init__(self):
        self.name = 'AUS'
        self.conf = DatasetConfig('AUS')
        self.case_collection = None

    #         self.subwordenc = load_file(self.conf.subwordenc_path)

    #     @staticmethod
    def load_all_reviews(self):
        """
        Returns:
            reviews: list of dicts
            item_to_reviews: dict, key=str (item id), value=list of dicts
        """
        case_collection = {}
        n = 0
        fnames = os.listdir(self.conf.raw_path)
        for fname in fnames:
            fpath = self.conf.raw_path + fname
            try:
                case_collection[fname] = load_file(fpath)
            except Exception as e:
                print(e)

        return case_collection, len(fnames)

    def get_data_loader(self, split='train', subset=None,
                        batch_size=2, shuffle=True, num_workers=4):
        """
        Return iterator over specific split in dataset
        """
        ds = AUSPytorchDataset(split=split, subset=subset)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader

    ####################################
    #
    # One off functions
    #
    ####################################
    def save_processed_splits(self):
        """
        Save train, val, and test splits. Splits are across items (e.g. a item is either in train, val, or test).
        Iterates over all reviews in the original dataset. Tries to get close to a 80-10-10 split.
        Args:
            review_max_len: int (maximum length in subtokens a review can be)
            item_min_reviews: int (min number of reviews a item must have)
            out_dir: str (path to save splits to, e.g. datasets/amazon_dataset/proccessed/)
        """
        sent_max_len = self.conf.sent_max_len
        item_min_sent = self.conf.item_min_sent
        item_max_sent = self.conf.item_max_sent
        item_min_catch = self.conf.item_min_catch
        item_max_catch = self.conf.item_max_catch

        print('Saving processed splits')
        if self.case_collection is None:
            self.case_collection, num_file = self.load_all_reviews()

        case_collection_len = num_file

        for key in self.case_collection.keys():
            self.case_collection[key]["num_sentence"] = len(self.case_collection[key]["sentences"])
            self.case_collection[key]["num_catchphrase"] = len(self.case_collection[key]["catchphrases"])
            self.case_collection[key]["max_sent_length"] = max(
                [len(sent) for sent in self.case_collection[key]["sentences"]])
            self.case_collection[key]["max_catch_length"] = max(
                [len(sent) for sent in self.case_collection[key]["catchphrases"]])

        print('Filtering case having more than {} sentence or less than {} sentence '.format(item_max_sent,
                                                                                             item_min_sent))
        case_collection_1 = {}
        for key in self.case_collection.keys():
            if self.case_collection[key]["num_sentence"] < item_max_sent and self.case_collection[key][
                "num_sentence"] > item_min_sent:
                case_collection_1[key] = self.case_collection[key]
        case_collection_len_1 = len(case_collection_1)

        print('Filtering case having more than {} catchphrase or less than {} catchphrase '.format(item_max_catch,
                                                                                                   item_min_catch))
        case_collection_2 = {}
        for key in case_collection_1.keys():
            if case_collection_1[key]["num_catchphrase"] < item_max_catch and case_collection_1[key][
                "num_sentence"] > item_min_catch:
                case_collection_2[key] = case_collection_1[key]

        case_collection_len_2 = len(case_collection_2)

        # # Note: we actually do more filtering in the Pytorch dataset class
        print('Filtering case having sentence longer than: {}'.format(sent_max_len))
        case_collection_3 = {}
        for key in case_collection_2.keys():
            if case_collection_2[key]["max_sent_length"] < sent_max_len:
                case_collection_3[key] = case_collection_2[key]

        case_collection_len_3 = len(case_collection_3)
        self.case_collection_len_final = case_collection_len_3

        # Calculate target amount of reviews per item
        print('Total number of reviews before filtering: {}'.format(case_collection_len))
        print('Total number of reviews after filtering: {}'.format(self.case_collection_len_final))

        self.case_collection = case_collection_3

        fname = sorted(list(self.case_collection.keys()))
        item_to_n = dict(zip(fname, range(len(fname))))
        n_to_item = {v: k for k, v in item_to_n.items()}

        # Construct splits
        n = self.case_collection_len_final
        n_tr, n_val, n_te = int(0.8 * n), int(0.1 * n), int(0.1 * n)
        cur_n_tr, cur_n_val, cur_n_te = 0, 0, 0
        split_to_item_to_n = {'train': {}, 'val': {}, 'test': {}}
        # In descending order of number of reviews per item
        for i, (item, n) in enumerate(sorted(item_to_n.items())):
            # once every ten items, save to val / test if we haven't yet hit the target number
            if (i % 10 == 8) and (cur_n_val < n_val):
                split = 'val'
                cur_n_val += 1
            elif (i % 10 == 9) and (cur_n_te < n_te):
                split = 'test'
                cur_n_te += 1
            else:
                split = 'train'
                cur_n_tr += 1

            out_fp = os.path.join(self.conf.processed_path, '{}/{}'.format(split, item))
            in_fp = os.path.join(self.conf.raw_path, '{}'.format(item))
            copy_file(in_fp, out_fp, verbose=False)

            split_to_item_to_n[split][item] = n
        print('Number of train reviews: {} / {}'.format(cur_n_tr, n_tr))
        print('Number of val reviews: {} / {}'.format(cur_n_val, n_val))
        print('Number of test reviews: {} / {}'.format(cur_n_te, n_te))

        # This file is used by AmazonPytorchDataset
        for split, item_to_n in split_to_item_to_n.items():
            out_fp = os.path.join(self.conf.processed_path, '{}/item-to-n.json'.format(split))
            save_file(item_to_n, out_fp)


class AUSPytorchDataset(Dataset):
    """
    Implements Pytorch Dataset
    One data point for model is n_docs reviews for one item. When training, we want to have batch_size items and
    sample n_docs reviews for each item. If a item has less than n_docs reviews, we sample with replacement
    (sampling with replacement as then you'll be summarizing repeated reviews, but this shouldn't happen right now
    as only items with a minimum number of reviews is used (50). These items and theiR reviews are selected
    in AmazonDataset.save_processed_splits().
    """

    def __init__(self, split=None, n_docs=None,
                 subset=None,
                 seed=0,
                 sample_reviews=True):
        """
        Args:
            split: str ('train', val', 'test')
            n_docs: int
            subset: float (Value in [0.0, 1.0]. If given, then dataset is truncated to subset of the businesses
            seed: int (set seed because we will be using np.random.choice to sample reviews if sample_reviews=True)
            sample_reviews: boolean
                - When True, __getitem_ will sample n_docs reviews for each item. The number of times a item appears
                in the dataset is dependent on uniform_items.
                - When False, each item will appear math.floor(number of reviews item has / n_docs) times
                so that almost every review is seen (with up to n_docs - 1 reviews not seen).
                    - Setting False is useful for (a) validation / test, and (b) simply iterating over all the reviews
                    (e.g. to build the vocabulary).
            item_max_reviews: int (maximum number of reviews a item can have)
                - This is used to remove outliers from the data. This is especially important if uniform_items=False,
                as there may be a large number of reviews in a training epoch coming from a single item. This also
                still matters when uniform_items=True, as items an outlier number of reviews will have reviews
                that are never sampled.
                - For the Amazon dataset, there are 11,870 items in the training set with at least 50 reviews
                no longer than 150 subtokens. The breakdown of the distribution in the training set is:
                    Percentile  |  percentile_n_reviews  |  n_items  |  total_revs
        """
        self.split = split
        self.subset = subset
        self.sample_reviews = sample_reviews

        self.ds_conf = DatasetConfig('AUS')

        # Set random seed so that choice is always the same across experiments
        # Especially necessary for test set (along with shuffle=False in the DataLoader)
        np.random.seed(seed)
        self.item_to_n = load_file(
            os.path.join(self.ds_conf.processed_path, '{}/item-to-n.json'.format(split)))
        self.idx_to_item = {}

        idx = 0
        for item, n_reviews in self.item_to_n.items():
            self.idx_to_item[idx] = item
            idx += 1

        self.n=len(self.item_to_n)
        # pdb.set_trace()

    def __getitem__(self, idx):
        """

        :param idx:
        :return:case_content
            {sentences: sentences
            catchphrases: catchphrases  }
        """
        # Map idx to item and load reviews

        item = self.idx_to_item[idx]
        fp = os.path.join(self.ds_conf.processed_path, '{}/{}'.format(self.split, item))
        case_content = load_file(fp)

        sentences =EOS_TOK.join(case_content["sentences"])
        catchphrases = EOC_TOK.join(case_content["catchphrases"])

        return sentences,catchphrases

    def __len__(self):
        return self.n

if __name__ == '__main__':
    ds = AUSDataset()

    ds.save_processed_splits()