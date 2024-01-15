This is the pytorch implementation of the paper (accpted by IEEE TCSVT 2023).

## Train and Test

download the Aster model from https://github.com/ayumiymk/aster.pytorch, Moran model from https://github.com/Canjie-Luo/MORAN_v2, 
CRNN model from https://github.com/meijieru/crnn.pytorch.

Change `TRAIN.VAL.rec_pretrained` in **src/configs/super_resolution.yaml** to your Aster model path, change `TRAIN.VAL.moran_pretrained` to your MORAN model path and 
change `TRAIN.VAL.crnn_pretrained` to your CRNN  model path.

Change `TRAIN.train_data_dir0` to your train data path.
Change `TRAIN.VAL.val_data_dir0` to your val data path.

- train with textzoom

`cd ./src/`

`python3 main.py --batch_size=30 --STN --mask --gradient --vis_dir='demo1'`

- test with textzoom

`python3 main.py --batch_size=1024 --test --test_data_dir='your-test-lmdb-dataset' --resume='your-model.pth' --STN --mask --gradient --vis_dir='vis'`

- demo with images

`python3 main.py --demo --demo_dir='./images/'  --resume='your-model.pth' --STN --mask`

If you have any question, please contact us without hesitation.

If you find TEAN useful in your research, please consider citing.

@ARTICLE{10102515,
  author={Shu, Rui and Zhao, Cairong and Feng, Shuyang and Zhu, Liang and Miao, Duoqian},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Text-Enhanced Scene Image Super-Resolution via Stroke Mask and Orthogonal Attention}, 
  year={2023},
  volume={33},
  number={11},
  pages={6317-6330},
  doi={10.1109/TCSVT.2023.3267133}}

