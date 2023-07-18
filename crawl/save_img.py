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
    driver.save_screenshot('/opt/ml/wine/crawl/'+error + '.png')

    
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


        
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
def write_data(write_file, datas):
    write_file = pd.concat([write_file,  pd.DataFrame(datas)], ignore_index=True)
    return write_file

def save_img(url, wine_id, img_folder):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            save_to = os.path.join(img_folder, f'{wine_id}.png')
            with open(save_to, 'wb') as f: f.write(response.content)
            return True
        else: return False

    except: return False

def get_img(driver, urls, done, failed, item2idx, img_folder):
    class_name = "mobile-column-3.tablet-column-3.desktop-column-2"

    


    i = 0
    for url in tqdm(urls):

        driver.get(url)
        selenium_scroll_down(driver)
        time.sleep(2)
        close_chat(driver)
        driver.switch_to.default_content() 

        try:
            img_pannel = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, class_name)))
            img = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.TAG_NAME, 'img')))
            img_url = img.get_attribute('src')
            
            wine_id = item2idx[url]
            result = save_img(img_url, wine_id, img_folder)
            
            if result == False:
                failed.add(url)
            else:
                done.add(url)
        
        except:
            pass
        i += 1
        
        time.sleep(4)
        if i % 500 == 0:
            with open(os.path.join(data_dir,'img_done.pkl'), 'wb') as f: pickle.dump(done, f)
            with open(os.path.join(data_dir,'img_failed.pkl'), 'wb') as f: pickle.dump(failed, f)


    with open(os.path.join(data_dir,'img_done.pkl'), 'wb') as f: pickle.dump(done, f)
    with open(os.path.join(data_dir,'img_failed.pkl'), 'wb') as f: pickle.dump(failed, f)


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
    
    current_script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script_path)

    data_dir = current_dir.replace('crawl','data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    img_folder = os.path.join(current_dir.replace('crawl','data'), 'images')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    try:
        with open(os.path.join(data_dir, 'img_done.pkl'), 'rb') as f: done  = pickle.load(f)
    except:
        img_done = set()

    try:
        with open(os.path.join(data_dir, 'img_failed.pkl'), 'rb') as f: done  = pickle.load(f)
    except:
        img_failed = set()


    feature_mapper_dir = current_dir.replace('crawl','code')
    with open(os.path.join(feature_mapper_dir, 'feature_map','item2idx.json'), 'r') as f:
        item2idx = json.load(f)

    urls = list(item2idx.keys())

    get_img(driver, urls, img_done, img_failed, item2idx, img_folder)
    