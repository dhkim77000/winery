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
    driver.save_screenshot('/opt/ml/wine/'+error + '.png')

    
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

def close_chat(driver):
    try:
        iframe = driver.find_element(By.ID, 'forethought-chat')
        driver.switch_to.frame(iframe)
        driver.find_element(By.CLASS_NAME, "css-8s03np").click()
        driver.switch_to.default_content()
        return 
    except Exception as e:
        return
    
#----------------------------------------------------------------------------------------------
def to_wishlist(driver):
    try:
        class_name = "user-profile-menu-item text-inline-block text-muted".replace(' ','.')
        wishlist = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, class_name))).get_attribute('href')
        driver.get(wishlist)
        time.sleep(4)
        return True
    except:
        return False

def find_all_likes(driver):
    print('----------------Finding Interactions')
    try:
        class_name = 'col-sm-8.user.user-profile-main.col-xs-12'
        area = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, class_name)))
    
        stop_count = 0
        pbar = tqdm(total=300)
        while stop_count < 3:
            button = "show-more".replace(' ','.')
            try: 
                WebDriverWait(area , 5).until(EC.visibility_of_element_located((By.CLASS_NAME, button))).click()
                time.sleep(3)
                selenium_scroll_down(driver)
                time.sleep(1)
            except:
                stop_count += 1
            pbar.update(1)
        wish_class = "activity-card"
        wishlists = area.find_elements(By.CLASS_NAME,wish_class)
        return wishlists
    except:
        return []
    
def rated_or_not(wishlist):
    rated_orna = "activity-rating.text-small.rating-section.activity-section.clearfix"
    try:
        wishlist.find_element(By.CLASS_NAME,rated_orna)
        return False
    except:
        return True
    
def get_item(wishlists):
    wine_url_class = 'link-muted.bold'
    
    items = []
    for wishlist in tqdm(wishlists):
        try:
            wine_url = wishlist.find_element(By.CLASS_NAME,wine_url_class).get_attribute('href')
            items.append(wine_url)
        except:
            continue
    
    return items

def get_implicit_feedback(driver, url):

    try:
        driver.get(url)
    except:
        driver = get_driver(chrome_options, url)
    
    
    selenium_scroll_down(driver)
    time.sleep(2)
    close_chat(driver)
    driver.switch_to.default_content() 

    driver.set_window_size(360, 1080)
    driver.execute_script("document.body.style.zoom='20%'")

    to_wishlist(driver)
    
    wishlists = find_all_likes(driver)
    return get_item(wishlists)



#------------------------------------------------------------------------------------------------
def main(driver, urls, done, df):
    
    for url in tqdm(urls):
        if url not in done:
    
            items = get_implicit_feedback(driver, url)
            print('----------------------Saving----------------------')
            inter = [{'user_url': url, 'wine_url':w} for w in items]

            df = write_data(df, inter)
            done.add(url)
            df.to_csv('/opt/ml/wine/data/implicit.csv', encoding = 'utf-8-sig',index= False)
            with open('/opt/ml/wine/data/user_done.pkl','wb') as f: pickle.dump(done,f)

            time.sleep(5)

    df = write_data(df, inter)
    df.to_csv('/opt/ml/wine/data/implicit.csv', encoding = 'utf-8-sig',index= False)
    with open('/opt/ml/wine/data/user_done.pkl','wb') as f: pickle.dump(done,f)
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


    with open('/opt/ml/wine/data/urls.json', 'r') as f: urls = json.load(f)

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

    user_urls = set()

    for i in tqdm(range(6)):

        if i == 0:
            url = pd.read_csv(f'/opt/ml/wine/data/review_df0.csv', encoding = 'utf-8-sig').loc[:,'user_url']
        else:
            url = pd.read_csv(f'/opt/ml/wine/data/review_df{i}.csv', encoding = 'utf-8-sig').loc[:,'user_url']
        url.dropna(inplace = True)

        user_urls = user_urls.union(set(list(url)))

    urls_for_me = split_list(list(user_urls), 5)[my_idx]

        

    try:
        with open('/opt/ml/wine/data/user_done.pkl', 'rb') as f: done  = pickle.load(f)
    except: done = set()

    try:
        df = pd.read_csv('/opt/ml/wine/data/implicit.csv', encoding = 'utf-8-sig')
    except:
        columns = ['user_url','wine_url']
        df = pd.DataFrame(columns = columns)
    print(len(urls_for_me))
    main(driver, urls_for_me, done, df)
    