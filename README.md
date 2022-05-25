# SequentialStyle

This is the offical implementation of the paper: "Y. Huang, Y. Liu, M. Jing, X. Zeng and Y. Fan, "Tear the Image into Strips for Style Transfer," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2021.3111515."

The paper can be found at: [SequentialStyle](https://ieeexplore.ieee.org/document/9537652)

Tensorflow 1.08-1.14

Train SequentialStyle example:

python train.py -is_training=True -style_name=./star.jpg -train_data_path=./content/ -vgg_model=./VGG16/vgg_16.ckpt -style_w=100

Test SequentialStyle example:

python train.py -model=./model_saved/40000/model.ckpt -test_content=./content.jpg -stylized_img=./stylized.jpg

please cite the paper as:

"Y. Huang, Y. Liu, M. Jing, X. Zeng and Y. Fan, "Tear the Image into Strips for Style Transfer," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2021.3111515."

@ARTICLE{9537652,
  author={Huang, Yujie and Liu, Yuhao and Jing, Minge and Zeng, Xiaoyang and Fan, Yibo},
  journal={IEEE Transactions on Multimedia}, 
  title={Tear the Image into Strips for Style Transfer}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2021.3111515}}
