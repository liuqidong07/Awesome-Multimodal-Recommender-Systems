# Multimodal Recommender Systems: A Survey

A collection of resources and papers of __Multimodal Recommender Systems__ (MRS).

🔥🔥 We will update the repo sustainably!

## 1. Our Survey

In our survey, we conclude the general MRS as an unified process, including Raw Feature Representation, Feature Interaction and Recommend Model. To face with the challenges contained in each procedure, we classify the existing works according to four branch of techniques, i.e., __Modality Encoder__, __Feature Interaction__, __Feature Enhancement__ and __Optimization__.

<img src="MRS.png" style="zoom: 33%;" />

More details can be seen in our survey.

## 2. Open-sourced Repositories

There are two open-sourced repositories for implementing multimodal recommender system models.

[MMRec](https://github.com/enoche/MMRec): A PyTorch benchmark, which implements 15 most recent MRS models. 

[Cornec](https://github.com/PreferredAI/cornac): A PyTorch framework, which implements more earlier MRS model.

## 3. Datasets

| Data           |       Field        | Modality |  Scale   |                             link                             |
| -------------- | :----------------: | :------: | :------: | :----------------------------------------------------------: |
| Tiktok         |    Micro-video     | V,T,M,A  |  726K+   |  [link](https://paperswithcode.com/dataset/tiktok-dataset)   |
| Kwai           |    Micro-video     |  V,T,M   |   1M+    |    [link](https://zenodo.org/record/4023390#.Y9YZ6XZBw7c)    |
| Movielens+IMDB |       Movie        |   V,T    | 100K-25M |      [link](https://grouplens.org/datasets/movielens/)       |
| Douban         | Movie, Book, Music |   V,T    |   1M+    | [link](https://github.com/FengZhu-Joey/GA-DTCDR/tree/main/Data) |
| Yelp           |        POI         | V,T,POI  |   1M+    |             [link](https://www.yelp.com/dataset)             |
| Amazon         |     E-commerce     |   V,T    |  100M+   | [link](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) |
| Book-Crossings |        Book        |   V,T    |   1M+    | [link](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) |
| Amazon Books   |        Book        |   V,T    |    3M    |        [link](https://jmcauley.ucsd.edu/data/amazon/)        |
| Amazon Fashion |      Fashion       |   V,T    |    1M    |        [link](https://jmcauley.ucsd.edu/data/amazon/)        |
| POG            |      Fashion       |   V,T    |   1M+    | [link](https://drive.google.com/drive/folders/1xFdx5xuNXHGsUVG2VIohFTXf9S7G5veq) |
| TMall          |      Fashion       |   V,T    |   8M+    |        [link](https://tianchi.aliyun.com/dataset/43)         |
| Taobao         |      Fashion       |   V,T    |   1M+    |        [link](https://tianchi.aliyun.com/dataset/52)         |
| Tianchi News   |        News        |    T     |   3M+    | [link](https://tianchi.aliyun.com/competition/entrance/531842/introduction) |
| MIND           |        News        |   V,T    |   15M+   |              [link](https://msnews.github.io/)               |
| Last.FM        |       Music        |  V,T,A   |  186K+   | [link](https://www.heywhale.com/mw/dataset/5cfe0526e727f8002c36b9d9/content) |
| MSD            |       Music        |   T,A    |   48M+   |       [link](http://millionsongdataset.com/challenge/)       |

Note: ’V’, ’T’, ’M’, ’A’ indicate the visual data, textual data, video data and acoustic data, respectively

## 4. Paper List

| Name | Paper                                                                                                                                                   |   Feature Interaction   | Feature Enhancement | Optimization |  Venue  |                  Code                  |
| :---: | ------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------: | :-----------------: | :----------: | :------: | :-------------------------------------: |
| MARank | [Multi-order attentive ranking model for sequential recommendation](https://aaai.org/ojs/index.php/AAAI/article/download/4516/4394) | Combined Attention | None | End-to-end | AAAI'19 | [link](https://github.com/voladorlu/MARank) |
| SAERS | [Explainable Fashion Recommendation: A Semantic Attribute Region Guided Approach](https://arxiv.org/pdf/1905.12862) | Fine-grained Attention | None | End-to-end | IJCAI'19 | N/A |
| MKR | [Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation](https://dl.acm.org/doi/pdf/10.1145/3308558.3313411) | Knowledge Graph | None | End-to-end | WWW'19 | [link](https://github.com/hwwang55/MKR) |
| UVCAN | [User-Video Co-Attention Network for Personalized Micro-video Recommendation](https://dl.acm.org/doi/pdf/10.1145/3308558.3313513) | Coarse-grained Attention | None | End-to-end | WWW'19 | N/A |
| VECF | [Personalized Fashion Recommendation with Visual Explanations based on Multimodal Attention Network](https://dl.acm.org/doi/pdf/10.1145/3331184.3331254)  |  Fine-grained Attention  |        None        |  End-to-end  | SIGIR'19 |                   N/A                   |
| NRPA | [NRPA: Neural Recommendation with Personalized Attention](https://dl.acm.org/doi/pdf/10.1145/3331184.3331371) | Combined Attention | None | End-to-end | SIGIR'19 | N/A |
| POG | [POG: Personalized Outfit Generation for Fashion Recommendation at Alibaba iFashion](https://dl.acm.org/doi/pdf/10.1145/3292500.3330652) | Fine-grained Attention | None | Two-step | KDD'19 | [link](https://github.com/wenyuer/POG) |
| MMGCN | [MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video](https://dl.acm.org/doi/pdf/10.1145/3343031.3351034) | User-item Graph | None | End-to-end | MM'19 | [link](https://github.com/weiyinwei/MMGCN) |
|  | [Learning Disentangled Representations for Recommendation](https://proceedings.neurips.cc/paper_files/paper/2019/file/a2186aa7c086b46ad4e8bf81e2a3a19b-Paper.pdf) | Other Fusion | DRL | End-to-end | NIPS'19 | [link](https://jianxinma.github.io/disentangle-recsys.html) |
| NOR | [Explainable Outfit Recommendation with Joint Outfit Matching and Comment Generation](https://ieeexplore.ieee.org/abstract/document/8669792/) | Fine-grained Attention | None | End-to-end | TKDE'19 | N/A |
| IRIS | [Interest-Related Item Similarity Model Based on Multimodal Data for Top-N Recommendation](https://ieeexplore.ieee.org/abstract/document/8618448/) | Other Fusion | None | End-to-end | Access'19 | N/A |
| BGCN | [Bundle Recommendation with Graph Convolutional Networks](https://dl.acm.org/doi/pdf/10.1145/3397271.3401198) | Item-item Graph | CL | End-to-end | SIGIR'20 | [link](https://github.com/cjx0525/BGCN) |
| DICER | [Content-Collaborative Disentanglement Representation Learning for Enhanced Recommendation](https://dl.acm.org/doi/pdf/10.1145/3383313.3412239) | Other Fusion | DRL | End-to-end | RecSys'20 | N/A |
| GRCN | [Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback](https://dl.acm.org/doi/pdf/10.1145/3394171.3413556) | Filtration | None | End-to-end | MM'20 | [link](https://github.com/weiyinwei/GRCN) |
| MKGAT | [Multi-modal Knowledge Graphs for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3340531.3411947) | Knowledge Graph + Filtration | None | End-to-end | CIKM'20 | N/A |
| MGAT | [MGAT: Multimodal Graph Attention Network for Recommendation](https://pdf.sciencedirectassets.com/271647/1-s2.0-S0306457320X00021/1-s2.0-S0306457320300182/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIGgxpjAAQQH1dspD3s4ED8Ub339%2FPYA4om4YWQFZ6njJAiAn70RxdrGas6d5YbgutMPKcfxsi62KGuqc7psCQWKEJiqyBQgYEAUaDDA1OTAwMzU0Njg2NSIMEe8JjCNTV9elaKiiKo8F6Qk2YTJ6RhCrPd%2B0DZsgnbgM%2BjNbZaNMnsCNzGzcv4VfU8S5pt3NZa1IjU39QMBnuXQliiWzKtS0flMI2rF8uhYnHaQhhCLEV%2FlnNJH%2F9Cy3iz%2Bm0jJgeWkYEPUEEdvo4xbeQErQZ%2F2Ov18BBYDuxvzdG8VOB8sL9soBMGpA7H4acEKPISftQ5XBAR8JbV72maAOpBromHI9KqsfppzxsUrBN%2Fj5jZBOxMhm66s68KmwsA3xNRgRCkbxtJx5%2B9psE7d52cv9rsnP3htY3RVf3w1cLP5Qxbc5AE5APzt8UHmDyRsE2dUs68CVC5vIYEl6%2FAjLQ13Q7%2FxTNSlo257mC7OWLr%2F9cjjlgxf%2BdO%2BaCPxRoacUB09fWvEd1iVcjScZsM6QyyE1TddO15IeAz3d7es7qQYXLUEjrKcFxdHB3mt%2FxcbkF9AWdWgwYvvnVvEyUpjBATb69flaRqLsWePnNERrW3IAeUKazMfSdIcseKe12rjBR7MgJ%2FSFpw7LZohgQZye7VTx7Dpvl4%2BeqYRw7qqSbk%2FKKK4Q8gYM6UTyGk07fjQJMB49Za6uhswbwNsXUkztMsdGWnqK8pEcWRb0vZoWzJdFCBRaAA5BGYtpp%2BZsTXr2EROcI6ZkU6mM2vzIdUa3wwaPAhBHOY%2BFNigV1in7pllgfa4A5Xy3l7zkLIuu5iJnsTIAX%2BvDznMxfFitxGml5ORn2aAwhJLqvlagN5Nhiz1ZwLV%2FecjCOQJrawCEGXzD8pbxpsQVkNlfWA%2BI%2FWcWqhQqp%2FmGWRF7SboTnMEM2SZ2gr8jo%2FoHYdnFkSA7FQZUKnWSAX1fikotCYnwEeYuMJ9xlXEn3aUbSUdT%2BHo%2FFeVAXYrg3dH%2BP3zaOzDojN2yBjqyAYGXewvWSfK5zxSbINPWljTOuzsj9y6bJyNEQR%2BVU%2BFfwkQ2BcmdShmmCFcWtxOPDzSzUKgZqkK30Zb%2BViYxQoOvApfS85iKC4BbkJKaRmsuyTIfNlv%2Bb1O%2FtlMPH4CHLc0SargKkf%2BhPDj2giIHLZjXNyq%2BP%2BoU0EozoZVzbGTIHNTCJb1ltpekSAoYniZOhQsE2AZVQep8huHTIHwSaM54vbgt8ViuuO4a3jRh70ahXjw%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240529T160502Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRIBZBNWQ%2F20240529%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=63f79397cbc2420ed845365cde1d6a03bf20669b7d68aff0e3a643b683139f3b&hash=d9d41f3c37f61ba06ef9c6635102351ff8537f4bb2f45c2da9aeed5b5d784b81&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0306457320300182&tid=spdf-0764f9b3-7ab7-4abd-b5cb-f663bebb4c4f&sid=bcfb68b52f3af04cc91a55439c2e0e1d6426gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=07095e5900500d565a&rr=88b7b9414a81105b&cc=cn) | User-item Graph + Fine-gained Attention | None | End-to-end | IPM'20 | N/A |
| SI-MKR | [An Enhanced Multi-Modal Recommendation Based on Alternate Training With Knowledge Graph Representation](https://ieeexplore.ieee.org/abstract/document/9264240) | Knowledge Graph | None | End-to-end | Access'20 | N/A |
| NOVA | [Noninvasive self-attention for side information fusion in sequential recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/16549) | Combined Attention | None | End-to-end | AAAI'21 | N/A |
| LATTICE | [Mining Latent Structures for Multimedia Recommendation](https://dl.acm.org/doi/pdf/10.1145/3474085.3475259) | Item-item Graph | None | End-to-end | MM'21 | [link](https://github.com/CRIPAC-DIG/LATTICE) |
| PMGT | [Pre-training Graph Transformer with Multimodal Side Information for Recommendation](https://dl.acm.org/doi/pdf/10.1145/3474085.3475709) | Item-item Graph + Fine-gained Attention | None | Two-step | MM'21 | N/A |
| VICTOR | [Understanding Chinese Video and Language via Contrastive Multimodal Pre-Training](https://dl.acm.org/doi/pdf/10.1145/3474085.3475431) | Fine-grained Attention | CL | Two-step | MM'21 | N/A |
| CDR | [Curriculum Disentangled Recommendation with Noisy Multi-feedback](https://proceedings.neurips.cc/paper/2021/file/e242660df1b69b74dcc7fde711f924ff-Paper.pdf) | Other Fusion | DRL | End-to-end | NIPS'21 | [link](https://github.com/forchchch/CDR) |
| MDR | [Multimodal Disentangled Representation for Recommendation](https://ieeexplore.ieee.org/abstract/document/9428193/) | Other Fusion | DRL | End-to-end | ICME'21 | N/A |
| CMBF | [CMBF: Cross-Modal-Based Fusion Recommendation Algorithm](https://www.mdpi.com/1424-8220/21/16/5275) | Coarse-grained Attention | None | End-to-end | Sensor'21 | N/A |
| DualGNN | [DualGNN: Dual Graph Neural Network for Multimedia Recommendation](https://ieeexplore.ieee.org/abstract/document/9662655) | User-item Graph | None | End-to-end | TMM'21 | [link](https://github.com/wqf321/dualgnn) |
| UMPR | [Recommendation by Users’ Multimodal Preferences for Smart City Applications](https://ieeexplore.ieee.org/abstract/document/9152003) | Other Fusion | None | End-to-end | TII'21 | N/A |
|      | [Multi-Modal Contrastive Pre-training for Recommendation](https://dl.acm.org/doi/pdf/10.1145/3512527.3531378)                                              | Coarse-grained Attention |         CL         |  End-to-end  | ICMR'22 |                   N/A                   |
| PAMD | [Modality Matches Modality: Pretraining Modality-Disentangled Item Representations for Recommendation](https://dl.acm.org/doi/pdf/10.1145/3485447.3512079) | Fine-gained Attention | DRL | End-to-end | WWW'22 | [link](https://github.com/hantengyue/PAMD) |
| SimGCL | [Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation](https://dl.acm.org/doi/pdf/10.1145/3477495.3531937) | User-item Graph | CL | End-to-end | SIGIR'22 | [link](https://github.com/Coder-Yu/QRec) |
| GHMFC | [Multimodal Entity Linking with Gated Hierarchical Fusion and Contrastive Training](https://dl.acm.org/doi/pdf/10.1145/3477495.3531867) | Knowledge Graph | CL | End-to-end | SIGIR'22 | [link](https://github.com/seukgcode/MEL-GHMFC) |
| MKGformer | [Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion](https://dl.acm.org/doi/pdf/10.1145/3477495.3531992) | Knowledge Graph + Fine-gained Attention | None | End-to-end | SIGIR'22 | [link](https://github.com/zjunlp/MKGformer) |
| MMGCL | [Multi-modal Graph Contrastive Learning for Micro-video Recommendation](https://dl.acm.org/doi/pdf/10.1145/3477495.3532027) | User-item Graph | CL | End-to-end | SIGIR'22 | N/A |
| MM-Rec | [MM-Rec: Multimodal News Recommendation](https://arxiv.org/pdf/2104.07407) | Fine-grained Attention | None | End-to-end | SIGIR'22 | N/A |
| CrossCBR | [CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation](https://dl.acm.org/doi/pdf/10.1145/3534678.3539229) | Item-item Graph | CL | End-to-end | KDD'22 | [link](https://github.com/mysbupt/CrossCBR) |
| Combo | [Combo-Fashion: Fashion Clothes Matching CTR Prediction with Item History](https://dl.acm.org/doi/pdf/10.1145/3534678.3539101) | Fine-grained Attention | CL | End-to-end | KDD'22 | [link](https://github.com/zhuchenxv/ComboFashion) |
| HCGCN | [Learning Hybrid Behavior Patterns for Multimedia Recommendation](https://dl.acm.org/doi/pdf/10.1145/3503161.3548119) | Item-item Graph | None | End-to-end | MM'22 | N/A |
| CKGC | [Cross-modal Knowledge Graph Contrastive Learning for Machine Learning Method Recommendation](https://dl.acm.org/doi/pdf/10.1145/3503161.3548273) | Knowledge Graph | CL | End-to-end | MM'22 | N/A |
|  MML  | [Multimodal Meta-Learning for Cold-Start Sequential Recommendation](https://dl.acm.org/doi/pdf/10.1145/3511808.3557101)                                   | Coarse-grained Attention |        None        |   Two-step   | CIKM'22 | [link](https://github.com/RUCAIBox/MML)  |
| MARIO | [MARIO: Modality-Aware Attention and Modality-Preserving Decoders for Multimedia Recommendation](https://dl.acm.org/doi/pdf/10.1145/3511808.3557387)      |  Fine-grained Attention  |        None        |  End-to-end  | CIKM'22 |                   N/A                   |
| MARIO | [MARIO: Modality-Aware Attention and Modality-Preserving Decoders for Multimedia Recommendation](https://dl.acm.org/doi/pdf/10.1145/3511808.3557387) | Fine-grained Attention | None | End-to-end | CIKM'22 | N/A |
| MMKGV | [Multi-modal Graph Attention Network for Video Recommendation](https://ieeexplore.ieee.org/abstract/document/9906399) | Knowledge Graph + Fine-gained Attention | None | End-to-end | CCET'22 | N/A |
| TESM | [A two-stage embedding model for recommendation with multimodal auxiliary information](https://www.sciencedirect.com/science/article/pii/S0020025521009270) | User-item Graph + Fine-gained Attention | None | Two-step | IS'22 | N/A |
| MICRO | [Latent Structure Mining With Contrastive Modality Fusion for Multimedia Recommendation](https://ieeexplore.ieee.org/abstract/document/9950351) | Item-item Graph | CL | End-to-end | TKDE'22 | [link](https://github.com/CRIPAC-DIG/MICRO) |
| DMRL | [Disentangled Multimodal Representation Learning for Recommendation](https://ieeexplore.ieee.org/abstract/document/9930669) | Fine-grained Attention | DRL | End-to-end | TMM'22 | [link](https://github.com/liufancs/DMRL) |
|  | [Implicit semantic-based personalized micro-videos recommendation](https://arxiv.org/pdf/2205.03297) | Fine-grained Attention | None | End-to-end | arXiv'22 | N/A |
|   VLSNR   | [VLSNR:Vision-Linguistics Coordination Time Sequence-aware News Recommendation](https://arxiv.org/pdf/2210.02946) |                    Combined Attention                    |        None         |  End-to-end  | arXiv'22  |       [link](https://github.com/Aaronhuang-778/V-MIND)       |
| BM3 | [Bootstrap Latent Representations for Multi-modal Recommendation](https://dl.acm.org/doi/pdf/10.1145/3543507.3583251) | User-item Graph + Other Fusion | CL | End-to-end | WWW'23 | [link](https://github.com/enoche/BM3) |
| MMMLP | [MMMLP: Multi-modal Multilayer Perceptron for Sequential Recommendations](https://dl.acm.org/doi/pdf/10.1145/3543507.3583378) | Other Fusion | None | End-to-end | WWW'23 | [link](https://github.com/Applied-Machine-Learning-Lab/MMMLP) |
| MMSSL | [Multi-Modal Self-Supervised Learning for Recommendation](https://dl.acm.org/doi/pdf/10.1145/3543507.3583206) | User-item Graph + Coarse-grained Attention | CL | End-to-end | WWW'23 | [link](https://github.com/HKUDS/MMSSL) |
| TMFUN | [Attention-guided Multi-step Fusion: A Hierarchical Fusion Network for Multimodal Recommendation](https://dl.acm.org/doi/pdf/10.1145/3539618.3591950) | Item-item Graph + Coarse-grained Attention | CL | End-to-end | SIGIR'23 | N/A |
| MCLN | [Multimodal Counterfactual Learning Network for Multimedia-based Recommendation](https://dl.acm.org/doi/pdf/10.1145/3539618.3591739) | Filtration | None | End-to-end | SIGIR'23 | [link](https://github.com/shuaiyangli/MCLN) |
| | [Enhancing Adversarial Robustness of Multi-modal Recommendation via Modality Balancing](https://dl.acm.org/doi/pdf/10.1145/3581783.3612337) | Filtration | None | End-to-end | MM'23 | N/A |
| MGCN | [MGCN: Multi-View Graph Convolutional Network for Multimedia Recommendation](https://dl.acm.org/doi/pdf/10.1145/3581783.3613915) | User-item Graph+Item-item Graph+Coarse-grained Attention | None | End-to-end | MM'23 | [link](https://github.com/demonph10/MGCN) |
| SGFD | [Semantic-Guided Feature Distillation for Multimodal Recommendation](https://dl.acm.org/doi/pdf/10.1145/3581783.3611886) | User-item Graph | None | Two-step | MM'23 | [link](https://github.com/HuilinChenJN/SGFD) |
| LATTICE | [A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation](https://dl.acm.org/doi/pdf/10.1145/3581783.3611943) | Filtration | None | End-to-end | MM'23 | [link](https://github.com/enoche/FREEDOM) |
| MMSR | [Adaptive Multi-Modalities Fusion in Sequential Recommendation Systems](https://dl.acm.org/doi/pdf/10.1145/3583780.3614775) | Item-item Graph + Combined-attention | None | End-to-end | CIKM'23 | [link](https://github.com/HoldenHu/MMSR) |
| M3Srec | [Multi-modal Mixture of Experts Representation Learning for Sequential Recommendation](https://dl.acm.org/doi/pdf/10.1145/3583780.3614978) | Other Fusion | CL | Two-step | CIKM'23 | [link](https://github.com/RUCAIBox/M3SRec) |
| MEGCF | [MEGCF: Multimodal Entity Graph Collaborative Filtering for Personalized Recommendation](https://dl.acm.org/doi/pdf/10.1145/3544106) | Filtration | None | End-to-end | TOIS'23 | [link](https://github.com/hfutmars/MEGCF) |
| SEM | [Disentangled Representation Learning for Recommendation](https://ieeexplore.ieee.org/abstract/document/9720218) | Other Fusion | DRL | End-to-end | TPAMI'23 | N/A |
| PromptMM | [PromptMM: Multi-Modal Knowledge Distillation for Recommendation with Prompt-Tuning](https://dl.acm.org/doi/pdf/10.1145/3589334.3645359) | User-item Graph | None | Two-step | WWW'24 | [link](https://github.com/HKUDS/PromptMM) |
| MG | [Mirror Gradient: Towards Robust Multimodal Recommender Systems via Exploring Flat Local Minima](https://dl.acm.org/doi/pdf/10.1145/3589334.3645553) | Filtration | None | End-to-end | WWW'24 | [link](https://github.com/Qrange-group/Mirror-Gradient) |
