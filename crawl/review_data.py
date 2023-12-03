from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
import time
import urllib.request
import os
import numpy as np
import pandas as pd
from urllib.parse import quote_plus          
from bs4 import BeautifulSoup as bs 
from xvfbwrapper import Xvfb
import time
from urllib.request import (urlopen, urlparse, urlunparse, urlretrieve)
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
import re
from selenium.webdriver.chrome.service import Service
import os 
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import pdb
import os
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import traceback         
from selenium.webdriver.common.proxy import Proxy, ProxyType
import csv
import requests
import json
import random
import psutil
import pickle
import warnings
import sys

def selenium_scroll_down(driver):
    SCROLL_PAUSE_SEC = 3
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_SEC)
        new_height = driver.execute_script("return document.body.scrollHeight")
        driver.find_element(By.TAG_NAME,'body').send_keys(Keys.CONTROL + Keys.HOME)
        time.sleep(1)
        if new_height == last_height: return 1
        last_height = new_height

def click(driver):
    try:
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
        content = driver.find_element(By.ID, "sp_message_iframe_737779")
        driver.switch_to.frame(content)
        driver.find_element(By.XPATH, '//*[@id="notice"]/div[3]/button').click()
        driver.switch_to.default_content()
        time.sleep(3)
        return True
    except Exception:
        return True


def get_driver(chrome_options, url):
    driver = None
    count = 0
    
    while (driver == None) and (count < 10):
            try:
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            except Exception:
                count = count + 1
                clean_up()
                if driver: driver.quit()
                continue

    connect = False
    while connect == False: 
        try:
            driver.get(url)
            driver.implicitly_wait(10)
            connect = True
        except Exception:
            del driver
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            continue
    return driver


def screenshot(driver, error):
    driver.save_screenshot('/home/dhkim/server_front/winery_AI/winery/'+error + '.png')

    
def reset_driver(driver, chrome_options, url):

    try :
        driver = get_driver(chrome_options, url)
        click(driver)
    except Exception:
        driver = get_driver(chrome_options, url)
        click(driver)
    #clean_up()
    return driver


def kill_process(name):
    try:
        for proc in psutil.process_iter():
            if proc.name() == name:
                proc.kill()
    except Exception:
        return

def clean_up():
    kill_process('chrome')
    kill_process('chromedriver') 
#------------------------------------------------------------------------------------------------


def click_more_review(driver):
    class_name="col mobile-column-6 tablet-column-8 desktop-column-9"
    try:
        review_tab = driver.find_element(By.CLASS_NAME, class_name.replace(' ','.'))
        review_tab.find_element(By.TAG_NAME, 'button').click()
        time.sleep(1)
        return True
    except Exception as e:
        screenshot(driver, f'close_{e}') 
        return False
    

    

def close_chat(driver):
    try:
        iframe = driver.find_element(By.ID, 'forethought-chat')
        driver.switch_to.frame(iframe)
        driver.find_element(By.CLASS_NAME, "css-8s03np").click()
        driver.switch_to.default_content()
        return 
    except Exception as e:
        return
    
#------------------------------------------------------------------------------------------------
def get_reaction(review):
    try:
        reaction_area_class = "communityReview__userActions--2RDK9"
        
        reaction_area = WebDriverWait(review, 2).until(EC.visibility_of_element_located((By.CLASS_NAME, reaction_area_class.replace(' ',''))))
        reaction_info = reaction_area.find_elements(By.TAG_NAME, 'a')
        
        if len(reaction_info) == 2:
            like = int(reaction_info[0].text)
            bad = int(reaction_info[1].text)
        elif len(reaction_info) == 1:
            like = int(reaction_info[0].text)
            bad = 0
        else: like, bad = 0, 0
  
    except:like, bad = 0, 0
        
    return like, bad
  
def get_user_info(review):
    try:
        user_info_class = "communityReview__textInfo--7SzS6"
        user_area = WebDriverWait(review, 3).until(EC.visibility_of_element_located((By.CLASS_NAME, user_info_class.replace(' ',''))))
        user_info = user_area.find_elements(By.TAG_NAME, 'a')

        if len(user_info) == 2:
            user_url = user_info[0].get_attribute('href')
            review_date = user_info[1].text
        elif len(user_info) == 1:
            user_url = user_info[0].get_attribute('href')
            review_date = None
        else:
            user_url, review_date = None, None
            
    except:
        user_url, review_date = None, None
        
    return user_url, review_date
    

def get_text_rate(review):
    try:
        text_class = "communityReview__reviewText--2bfLj"
        text = WebDriverWait(review, 2).until(EC.visibility_of_element_located((By.CLASS_NAME, text_class.replace(' ','')))).text
    except: text = None
    
    try:
        rating_class = "userRating_ratingNumber__cMtKU"
        WebDriverWait(review, 2).until(EC.visibility_of_element_located((By.CLASS_NAME, rating_class.replace(' ',''))))
        rating = float(review.find_element(By.CLASS_NAME, rating_class.replace(' ','')).text)
    except: rating = None
    
    return text, rating   
#------------------------------------------------------------------------------------------------
def get_review_info(review, stop_count):
    data = {}
    text, rating = get_text_rate(review)
    user_url, review_date = get_user_info(review)
    if user_url == None:
        stop_count += 1
    like, bad = get_reaction(review)
    data['text'] = text
    data['rating'] = rating
    data['user_url'] = user_url
    data['date'] = review_date
    data['like'] = like
    data['bad'] = bad
    return data, stop_count

def get_all_reviews(reviews, wine_url):

    review_datas = []
    stop_count = 0
    for review in tqdm(reviews):

        review_dict, stop_count = get_review_info(review, stop_count)
        review_dict['wine_url'] = wine_url
        review_datas.append(review_dict)
        time.sleep(0.2)
        if stop_count > 5: break
    
    return review_datas
   
def find_all_reviews(driver):
    stop_count = 0
    prv = 0

    print('-----Finding-----')
    pbar = tqdm(total=100)

    while stop_count < 5:
        ActionChains(driver).scroll_by_amount(0, 10000).perform()
        review_area = driver.find_element(By.CLASS_NAME, 'allReviews__reviews--EpUem'.replace(' ',''))
        reviews = driver.find_elements(By.CLASS_NAME, "communityReviewItem__reviewCard--1RupJ".replace(' ',''))
        if len(reviews) == prv:
            stop_count += 1
            time.sleep(2)
        else:
            stop_count = 0
        time.sleep(0.3)
        prv = len(reviews)
        if prv >= 500: break
        pbar.update(1)
        print(prv)
    pbar.close()
    print(f'-----Find {len(reviews)} reviews-----')

    return reviews
#------------------------------------------------------------------------------------------------
def get_stars(driver):
    try:
        class_name = 'RatingsFilter__pill--2V08n'
        stars = driver.find_element(By.CLASS_NAME, class_name)
        class_name = 'RatingsFilter__container--kWVlc'
        stars = stars.find_elements(By.CLASS_NAME, class_name)
        return stars
    except: return None

def wine_interaction(driver, url):
    try:
        driver.get(url)
    except:
        driver = get_driver(chrome_options, url)
    
    selenium_scroll_down(driver)
    time.sleep(2)
    close_chat(driver)
    driver.switch_to.default_content() 
    for _ in range(3):
        ActionChains(driver).scroll_by_amount(0, 1000).perform()
        time.sleep(0.5)

    recent_class ="anchor_anchor__m8Qi- menu__menuItem--1aKOP".replace(' ','.')

    if click_more_review(driver):
        
        driver.set_window_size(360, 1080)
        driver.execute_script("document.body.style.zoom='20%'")
        #stars = get_stars(driver)
        try: driver.find_element(By.CLASS_NAME, recent_class).click()
        except: 1

        reviews = find_all_reviews(driver)
        review_data = get_all_reviews(reviews, url)
        
        print("-------Finding under rating 3-------")
        
        return review_data
    
    else: return []
#------------------------------------------------------------------------------------------------
def main(driver, urls, done, df):
    
    for url in tqdm(urls):
        if url not in done:
    
            review = wine_interaction(driver, url)
            print('----------------------Saving----------------------')
            df = write_data(df, review)
            done.add(url)
            df.to_csv('/home/dhkim/server_front/winery_AI/winery/data/review_df.csv', encoding = 'utf-8-sig',index= False)
            with open('/home/dhkim/server_front/winery_AI/winery/data/review_done.pkl','wb') as f: pickle.dump(done,f)

            time.sleep(5)

    df = write_data(df, review)
    df.to_csv('/home/dhkim/server_front/winery_AI/winery/data/review_df.csv', encoding = 'utf-8-sig',index= False)
    with open('/home/dhkim/server_front/winery_AI/winery/data/review_done.pkl','wb') as f: pickle.dump(done,f)
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
def write_data(write_file, datas):
    write_file = pd.concat([write_file,  pd.DataFrame(datas)], ignore_index=True)
    return write_file


if __name__ == '__main__':

    vdisplay = Xvfb(width=1920, height=1080)
    vdisplay.start()
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-setuid-sandbox')
    #chrome_options.add_argument('--remote-debugging-port=9222')
    chrome_options.add_argument('--disable-dev-shm-usage')

    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument('--incognito')
    #mobile_emulation = { "deviceName" : "iPhone X" }
    #chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--allow-running-insecure-content')
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument("--start-maximized")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36'
    chrome_options.add_argument(f'user-agent={user_agent}')
    os.environ['WDM_LOG_LEVEL'] = '0'
    os.environ['WDM_LOG'] = "false"
  
    
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    driver = get_driver(chrome_options, 'https://www.vivino.com/US-CA/en/')

    my_idx = int(sys.argv[-1])


    with open('/home/dhkim/server_front/winery_AI/winery/data/urls.json', 'r') as f: urls = json.load(f)

    def split_list(lst, n):
        # Calculate the length of each sublist
        sublist_length = len(lst) // n
        # Determine the remaining elements
        remaining_elements = len(lst) % n
        # Initialize the starting index
        index = 0
        # Create sublists
        sublists = []
        for i in range(n):
            # Calculate the sublist size
            size = sublist_length + (1 if i < remaining_elements else 0)
            # Extract sublist from the original list
            sublist = lst[index:index+size]
            # Add the sublist to the result
            sublists.append(sublist)
            # Update the starting index for the next sublist
            index += size
        return sublists

    urls_for_me = split_list(urls, 5)[my_idx]

    try:
        with open('/home/dhkim/server_front/winery_AI/winery/data/review_done.pkl', 'rb') as f: done  = pickle.load(f)
    except: done = set()

    try:
        df = pd.read_csv('/home/dhkim/server_front/winery_AI/winery/data/review_df.csv', encoding = 'utf-8-sig')
    except:
        columns = ['user_url','rating','date','like','bad', 'text','wine_url']
        df = pd.DataFrame(columns = columns)

    main(driver, urls_for_me, done, df)
    