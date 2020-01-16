# training model using tf.data.Dataset

## Dataset directory
### 
```sh
└─data
    └─RAF_single
       ├─test
       └─train
         ├─angry
         │   img1.jpg
         │   img2.jpg
         │   img3.jpg
         │   ...
         ├─disgusted
         ├─fearful
         ├─happy
         ├─neutral
         ├─sad
         └─surprised          
```

## load all images & label
```python
def get_label(path):
    label = path.split('/')[-2]
    return labels_dict[label]
    
    
def onehot_encode_label(path):
    onehot_label = unique_label_names == get_label(path)    
    onehot_label = onehot_label.astype(np.uint8)
    return onehot_label
    
    
data_list = glob('data/RAF_single_original/train/*/*.*')

labels_dict = {'angry':0, 'disgusted':1, 'fearful':2, 'happy':3, 'neutral':4, 'sad':5, 'surprised':6}
label_name_list = os.listdir('data/RAF_single_original/train')

# label onehot-encoding
label_list = [onehot_encode_label(path).tolist() for path in data_list]

```

## make dataset
```python
def load_image(path):        
    # load image
    image_string = tf.io.read_file(path)    
    
    # decode a JPEG-encoded image to a uint8 tensor
    image = tf.image.decode_jpeg(image_string, channels=3)    

    # convert image to dtype & scaling values between 0 and 1
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # resize image
    image = tf.image.resize(image, [100, 100])
    
    return image
    
    
dataset = tf.data.Dataset.from_tensor_slices((data_list, label_list))
dataset = dataset.map(lambda data_list, label_list: (load_image(data_list), label_list))

# If you want to shuffle, buffer_size > the number of samples in dataset
dataset = dataset.shuffle(buffer_size=int(len(data_list) + 1))
dataset = dataset.batch(32)
dataset = dataset.prefetch(32)
dataset = dataset.repeat()
```

## Simple CNN
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, 3, 1, padding='same', activation='relu', input_shape=(100, 100, 3)))
model.add(tf.keras.layers.Conv2D(32, 3, 1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(128, 3, 1, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(128, 3, 1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(256, 3, 1, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(256, 3, 1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='softmax'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=['acc'])

model.fit(dataset, epochs=100, steps_per_epoch=int(len(data_list) / batch_size))
```
