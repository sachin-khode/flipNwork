from selenium import webdriver
from save_images import make_directory, save_images, save_data_to_csv
from scrap_images import scrap_image_url
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd

# creating an instance of google chrome
DRIVER_PATH = '/home/sk-ji/Desktop/chromedriver_linux64/chromedriver'

# to run chrome in a headfull moe(like regular chrome)
driver = webdriver.Chrome(executable_path=DRIVER_PATH)
current_page_url = driver.get("https://www.flipkart.com/clothing-and-accessories/topwear/tshirt/men-tshirt/pr?sid=clo,ash,ank,edy&otracker=categorytree&otracker=nmenu_sub_Men_0_T-Shirts")
#current_page_url = driver.get("https://www.flipkart.com/clothing-and-accessories/topwear/shirt/men-shirt/casual-shirt/pr?sid=clo,ash,axc,mmk,kp7&otracker=categorytree&otracker=nmenu_sub_Men_0_Casual%20Shirts")
DIRNAME = "Taining"

# print("yaha tak chal raha hain")
make_directory(DIRNAME)
# make_directory2()
# print("yaha pe break ho raha hain")

start_page = 3
total_pages = 5

# Scraping the pages

for page in range(start_page, total_pages + 1):
    try:
        product_details = scrap_image_url(driver=driver)
        print("Scraping Page {0} of {1} pages".format(page, total_pages))

        #page_value = driver.find_element_by_xpath("//a[@class='_2Xp0TH fyt9Eu']").text
        page_value = driver.find_element_by_xpath("//a[@class='_2Xp0TH fyt9Eu']").text
        print("The current page scraped is {}".format(page_value))

        print("error is here")
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
        '''save_data_to_csv(data=product_details, filename='Men_T-Shirt.csv')'''
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
