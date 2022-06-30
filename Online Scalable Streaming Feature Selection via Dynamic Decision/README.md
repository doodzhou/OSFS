@article{Zhou_TKDD

author = {Zhou, Peng and Zhao, Shu and Yan, Yuanting and Wu, Xindong},

title = {Online Scalable Streaming Feature Selection via Dynamic Decision},

year = {2022},

issue_date = {October 2022},

publisher = {Association for Computing Machinery},

address = {New York, NY, USA},

volume = {16},

number = {5},

issn = {1556-4681},

url = {https://doi.org/10.1145/3502737},

doi = {10.1145/3502737},

abstract = {Feature selection is one of the core concepts in machine learning, which hugely impacts the model’s performance. For some real-world applications, features may exist in a stream mode that arrives one by one over time, while we cannot know the exact number of features before learning. Online streaming feature selection aims at selecting optimal stream features at each timestamp on the fly. Without the global information of the entire feature space, most of the existing methods select stream features in terms of individual feature information or the comparison of features in pairs. This article proposes a new online scalable streaming feature selection framework from the dynamic decision perspective that is scalable on running time and selected features by dynamic threshold adjustment. Regarding the philosophy of “Thinking-in-Threes”, we classify each new arrival feature as selecting, discarding, or delaying, aiming at minimizing the overall decision risks. With the dynamic updating of global statistical information, we add the selecting features into the candidate feature subset, ignore the discarding features, cache the delaying features into the undetermined feature subset, and wait for more information. Meanwhile, we perform the redundancy analysis for the candidate features and uncertainty analysis for the undetermined features. Extensive experiments on eleven real-world datasets demonstrate the efficiency and scalability of our new framework compared with state-of-the-art algorithms.},

journal = {ACM Trans. Knowl. Discov. Data},

month = {mar},

articleno = {87},

numpages = {20},

keywords = {Feature selection, scalable feature selection, three-way decision, feature streams}

}
