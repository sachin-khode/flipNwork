from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join


# In[28]:


img_width = 150
img_height = 150

train_data_dir = 'jeans'
validation_data_dir = 'trousers'
train_samples = 120
validation_samples = 30
epochs = 5
batch_size = 20


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[29]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[30]:


test_datagen = ImageDataGenerator(rescale= 1. / 255)


# In[31]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[32]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[33]:


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[34]:


model = Sequential()


# In[35]:


model.add(Conv2D(32,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[36]:


model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[37]:


model.add(Conv2D(64,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[38]:


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[39]:


model.summary()


# In[40]:


model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
              metrics=['accuracy'])


# In[43]:

print("running here")
model.fit_generator(train_generator,
                    steps_per_epoch=train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_samples // batch_size)

print("running 2")
# In[45]:


model.save_weights('first_try.h5')


# In[46]:


img_pred = image.load_img('/home/sk-ji/Cont_ent/dog vs cat/Keras_Deep_Learning-master/image_data/test/236.jpg',target_size=(150,150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred,axis=0)


# In[47]:


rslt = model.predict(img_pred)
print(rslt)
if rslt [0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    


# In[48]:


print(prediction)


# In[49]:


# pwd


# In[ ]:




