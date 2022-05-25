# SequentialStyle

Train SequentialStyle example:

python train.py -is_training=True -style_name=./star.jpg -train_data_path=./content/ -vgg_model=./VGG16/vgg_16.ckpt -style_w=100

Test SequentialStyle example:

python train.py -model=./model_saved/40000/model.ckpt -test_content=./content.jpg -stylized_img=./stylized.jpg
