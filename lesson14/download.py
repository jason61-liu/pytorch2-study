#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  shiyiliu
# @Date:  2025/6/14  20:46
# @File:  download.py
# @Project:  pytorchdemo
# @Software:  PyCharm

from torchaudio import datasets

datasets.SPEECHCOMMANDS(
    root="../dataset/SpeechCommands",
    url="speech_commands_v0.02",
    folder_in_archive="SpeechCommands",
    download=True,
)
