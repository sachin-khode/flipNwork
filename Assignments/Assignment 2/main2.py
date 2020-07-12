from selenium import webdriver
from save_images import make_directory, save_images, save_data_to_csv
from scrap_images import scrap_image_url
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
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

aaa_trouser = 'https://www.flipkart.com/search?q=trousers&sid=clo%2Cvua%2Cmle%2Clhk&as=on&as-show=on&otracker=AS_QueryStore_OrganicAutoSuggest_3_8_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_3_8_na_na_na&as-pos=3&as-type=RECENT&suggestionId=trousers%7CMen%27s+Trousers&requestId=bcf3bdc7-28b2-4be9-b2e2-2a5c698eb6ab&as-searchtext=trousers'
aaa_jeans = 'https://www.flipkart.com/search?q=jeans&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off'
urls = [aaa_jeans, aaa_trouser]
for idx, scrpr_url in enumerate(urls):
    # creating an instance of google chrome
    DRIVER_PATH = '/home/sk-ji/Desktop/chromedriver_linux64/chromedriver'

    # to run chrome in a headfull moe(like regular chrome)
    driver = webdriver.Chrome(executable_path=DRIVER_PATH)
    current_page_url = driver.get(scrpr_url)
    #current_page_url = driver.get("https://www.flipkart.com/clothing-and-accessories/topwear/shirt/men-shirt/casual-shirt/pr?sid=clo,ash,axc,mmk,kp7&otracker=categorytree&otracker=nmenu_sub_Men_0_Casual%20Shirts")
    if idx == 0:
        DIRNAME = "jeans"
    else:
        DIRNAME = "trousers"
    #DIRNAME = "Men_Shirt"
    make_directory(DIRNAME)

    start_page = 2
    total_pages = 6

    # Scraping the pages

    for page in range(start_page, total_pages + 1):
        try:
            product_details = scrap_image_url(driver=driver)
            print("Scraping Page {0} of {1} pages".format(page, total_pages))

            #page_value = driver.find_element_by_xpath("//a[@class='_2Xp0TH fyt9Eu']").text
            page_value = driver.find_element_by_xpath("//a[@class='_2Xp0TH fyt9Eu']").text
            print("The current page scraped is {}".format(page_value))

            # Downloading the images
            save_images(data=product_details, dirname=DIRNAME, page=page)
            print("Scraping of page {0} done".format(page))

            # Moving to the next page
            print("Moving the next page")
            button_type = driver.find_element_by_xpath("//div[@class='_2zg3yZ']//a[@class='_3fVaIS']//span").get_attribute('innerHTML')

            if button_type == 'Next':
                driver.find_element_by_xpath("//a[@class='_3fVaIS']").click()
            else:
                driver.find_element_by_xpath("//a[@class='_3fVaIS'][3]").click()

            #new_page = driver.find_element_by_xpath("//a[@class='_2Xp0TH fyt9Eu']").text
            new_page = driver.find_element_by_xpath("//a[@class='_2Xp0TH fyt9Eu']").text

            print("The new page is {}".format(new_page))

        except StaleElementReferenceException as Exception:

            print("We are facing an exception ")

            exp_page = driver.find_element_by_xpath("//a[@class='_2Xp0TH fyt9Eu']").text
            print("The page value at the time of exception is {}".format(exp_page))

            value = driver.find_element_by_xpath("//a[@class='_2Xp0TH fyt9Eu']")
            link = value.get_attribute('href')
            driver.get(link)

            product_details = scrap_image_url(driver=driver)
            print("Scraping page {0} of {1} pages".format(page, total_pages))

            page_value = driver.find_element_by_xpath("//a[@class='_2Xp0TH fyt9Eu']").text
            print("Scraping of page {0} done".format(page_value))

            # downloading the images
            save_images(data=product_details, dirname=DIRNAME, page=page)
            print("Scraping of page {0} done".format(page))

            # saving the data into csv file
            # save_data_to_csv(data=product_details, filename='Men_T-Shirt.csv')
            #save_data_to_csv(data=product_details, filename='Men_Shirt.csv')
            # Moving to the next page
            print("Moving to the next page")

            button_type = driver.find_element_by_xpath("//div[@class='_2zg3yZ']//a[@class='_3fVaIS']//span").get_attribute('innerHTML')

            if button_type == 'Next':
                driver.find_element_by_xpath("//a[@class='_3fVaIS']").click()
            else:
                driver.find_element_by_xpath("//a[@class='_3fVaIS'][2]").click()

            new_page = driver.find_element_by_xpath("//a[@class='_2Xp0TH fyt9Eu']").text
            print("The new page is {}".format(new_page))


paths_for_training = ['jeans', 'trousers']

for paths in paths_for_training:

    img_width = 150
    img_height = 150

    train_data_dir = paths
    validation_data_dir = paths
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


    test_datagen = ImageDataGenerator(rescale=1. / 255)

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


    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # In[36]:


    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # In[37]:


    model.add(Conv2D(64, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

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


    img_pred = image.load_img('/home/sk-ji/Cont_ent/dog vs cat/Keras_Deep_Learning-master/image_data/test/236.jpg',
                              target_size=(150, 150))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)

    # In[47]:


    rslt = model.predict(img_pred)
    print(rslt)
    if rslt[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    # In[48]:


    print(prediction)

# In[49]:


# pwd


# In[ ]:




