from ROI import cap_save_image, get_rotate, rotate_times, record_times
from Tensor import get_files, get_tensor, get_dataset, train

print('训练自己的数据集，按r录取一张石头图像，按p录取一张布图像，按s录取一张剪刀图像。每个需要',
      record_times, '次, Esc结束\n' +
      '注意:小写，按键不是输入在控制台上，而是是在摄像头显示界面上直接按键即可')
file_path = 'Hand'

flag = cap_save_image()
if flag == 1:
    get_rotate(rotate_times)

    image_list, label_list = get_files(file_path)
    image_tensor, label_tensor = get_tensor(image_list, label_list)
    train_data = get_dataset(image_tensor, label_tensor)
    train_data = train_data.shuffle(1000).batch(16)     # 16x28x28x3
    train(train_data)
    print('Train完成\n')
else:
    print('图像录取未完成，请重新录取\n')

