import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import datetime
import glob


def make_dataset(start_time, end_time, img_dir, csv_dir, time_steps, img_interval=10, data_interval=1, minmax=[[0, 1], [0, 10], [0, 100]], BATCH_SIZE=128, BUFFER_SIZE=15000):
    """
    其他说明：输出的数据标签数据为202001011050，time_steps=2为例，输入历史数据为10：20,10：30，10:40的图片和
    10:11-10:20,10:21:-10:30,10:31-10:40的数据，输出结果为10:41-10:50的结果
    """

    def process_dataset(input_data, label_imgs, output_data, label_date):
        num = len(label_imgs)
        imgs = tf.TensorArray(size=num, dtype=tf.float32)
        for i in range(num):
            img = tf.io.read_file(label_imgs[i])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (288, 288))
            # img = tf.reshape(img, shape=(1, *img.shape))
            imgs.write(i, img)
        # tf.stack, tf.concat dont work when using map
        img = imgs.stack()
        img = img / 255.
        return input_data, img, output_data, label_date


    # process img images
    imgs_dir = glob.glob(os.path.join(img_dir, "*.jpg"))
    # all img names with date
    imgs_date = [dir.split('\\')[-1][:12] for dir in imgs_dir]
    date_img_range = pd.date_range(start_time, end_time, freq=str(img_interval)+'T').map(
        lambda x: x.to_pydatetime().strftime('%Y%m%d%H%M%S'))
    date_data_range = pd.date_range(start_time, end_time, freq=str(data_interval)+'T').map(
        lambda x: x.to_pydatetime().strftime('%Y%m%d%H%M%S'))
    # str type, valid img names in (start_time, end_time)
    new_imgs = [date for date in imgs_date if date in date_img_range]

    # process csv data
    data = pd.read_csv(csv_dir)
    # make new dataframe
    data_date = (data.iloc[:, 0] + ' ' + data.iloc[:, 1]).map(
        lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M').strftime('%Y%m%d%H%M%S'))
    data = data.iloc[:, 2:].set_index(pd.Series(data_date))
    temp_df = pd.DataFrame(index=date_data_range)
    for col in data.columns:
        temp_df[col] = data[col]
    data = temp_df.values[1:, :]
    minmax = np.array(minmax)
    minimum, maximum = minmax[:, 0], minmax[:, 1]
    data = (data - minimum) / (maximum - minimum)
    dim1, n_feature = int(data.shape[0]/10), data.shape[1]
    # data[:20, :] = np.arange(20*n_feature).reshape(20, n_feature)
    data = data.reshape(dim1, 10, n_feature).reshape(dim1, 10 * n_feature, order='F')
    data = pd.DataFrame(data, index=pd.Series(date_img_range[1:]),
                        columns=[f'f_{i}_t_{j}' for i in range(n_feature) for j in range(10)])
    data = pd.concat([data.shift(i).rename(
                        columns=lambda x: f's_{i}_' + x) for i in range(time_steps+1)], axis=1)
    data = data.dropna(axis=0, how='any')
    # check whether the imgs are valid for the forecast day
    all_data = pd.DataFrame()
    label_imgs, label_date = [], []
    for i in range(data.shape[0]):
        date = data.index[i]
        checks = [datetime.datetime.strptime(date, '%Y%m%d%H%M%S') - datetime.timedelta(minutes=10 * i)
                  for i in range(1, time_steps + 1)]
        checks = [date.strftime('%Y%m%d%H%M%S') for date in checks]
        if all([check in new_imgs for check in checks]):
            all_data = all_data.append(data.loc[date])
            #                                [10*n_feature:])
            # output_data = output_data.append(data.loc[date][:10])
            label_imgs.append([os.path.join(imgs_dir[0][:-24], date+imgs_dir[0][-10:])
                               for date in checks])
            label_date.append(date)
    all_data = all_data.to_numpy()
    all_data = all_data.reshape(all_data.shape[0], time_steps+1, 10*n_feature)
    input_data = all_data[:, 1:, :]
    output_data = all_data[:, 0, :10]

    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
    output_data = tf.convert_to_tensor(output_data, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((input_data, label_imgs, output_data, label_date))
    dataset = dataset.map(process_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset



if __name__ == "__main__":
    img_dir = r'C:\Users\26059\Desktop\img'
    csv_dir = r'C:\Users\26059\Desktop\data.csv'
    start = '2018-01-02 00:00:00'
    end = '2018-01-03 00:00:00'
    dataset = make_dataset(start, end, img_dir, csv_dir, 3, minmax=[[0, 1], [0, 10], [0, 100]], BATCH_SIZE=128, BUFFER_SIZE=15000)
    for epoch in range(500):
        for a, b, c, d in dataset:
            pass
        print(f'epoch {epoch}')
