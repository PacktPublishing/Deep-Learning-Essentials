#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# =============================================================================
"""
This code is using existing word2vec model. It is for
Chapter 5: Natural language processing - vector representation
"""

from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format(
    './GoogleNews-vectors-negative300.bin', binary=True)
similar_words = model.wv.most_similar(
    positive=['woman', 'king'], negative=['man'], topn=5)
print similar_words