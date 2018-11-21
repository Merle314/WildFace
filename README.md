# WildFace
Face Recognition in the wild
first convert vggface2 dataset to tfrecord to accelerate training
```python
python /tools/convert_vggface2.py
```
then convert lfw to npy file to accelerate data loading
```python
python /tools/lfw_npy.py
```
at last train the model use your own dataset
```python 
python train_tfrecord.py
'''
