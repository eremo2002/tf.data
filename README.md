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
