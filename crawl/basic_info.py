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
    driver.save_screenshot('/home/dhkim/vivino/'+error + '.png')

    
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
def get_basic_info(driver, url):

    driver.get(url)
    selenium_scroll_down(driver)
    time.sleep(2)
    close_chat(driver)
    driver.switch_to.default_content() 


    class_name = "wineFacts__wineFacts--2Ih8B"
    
    dic = {}
    dic['url'] = url
    try: 
        infos = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, class_name)))
        infos = infos.find_elements(By.TAG_NAME, 'tr')
        
        for info in infos:
            category, data = info.text.split('\n')
            if category == 'Grapes':
                dic[category.lower()] = [grape.strip().replace(' ', '_') for grape in data.lower().replace('100%','').split(',')]
            elif category == 'Region':
                data = [r.strip().replace(' ','_') for r in data.split('/')]
                dic['country'] = data[0]
                for i in range(1, len(data)):
                    dic[f'region{i}'] = data[i]
            else:
            
                dic[category.lower()] = data.lower().replace('100%','')

    except Exception as e: 
        print(e)
        return {}
    
    return dic
    
    

def get_pairings(driver):
    result = []
    class_name = "foodPairing__foodContainer--1bvxM"
    class_name2 = class_name2 = "anchor_anchor__m8Qi- foodPairing__imageContainer--2CtYR"
    try:
        pairings = driver.find_element(By.CLASS_NAME, class_name.replace(' ','.'))
        pairings = pairings.find_elements(By.CLASS_NAME, class_name2.replace(' ','.'))
        for p in pairings: result.append(p.text)
        return result
    except Exception as e:
        screenshot(driver, f'pairing_{e}') 
        return []
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
def get_parent_note(driver):
    class_name = "tasteReviews__capitalizedTastes--29J9-"
    class_name2 = "tasteReviews__mentionsText--1XXa4"
    try:
        note = driver.find_element(By.CLASS_NAME, class_name.replace(' ','.')).text
        num = driver.find_element(By.CLASS_NAME, class_name2.replace(' ','.')).text.split(' ')[0]
        return note, int(num)
    except Exception as e: 
        screenshot(driver, f'pnote_{e}') 
        return None, None


def get_tasting_notes(driver):
    parent_info = {}
    child_info = {}
    class_name =  "tasteReviews__keywords--3qDTs"
    parent, num = get_parent_note(driver)
    parent_info[parent] = parent
    parent_info[f'{parent}_count'] = num
    try:
        while True:
            taste_notes = driver.find_element(By.CLASS_NAME, class_name.replace(' ','.'))
            taste_notes = taste_notes.find_elements(By.TAG_NAME, 'a')
            for note in taste_notes:
                count_note = note.text.split('\n')
                if len(count_note) == 2: 
                    child_info[count_note[1]] = int(count_note[0].replace('k','000'))
            if click_next(driver) != True:break
            time.sleep(1)
        close_note(driver)
        return parent_info, child_info
    except Exception as e:
        screenshot(driver, f'tastenote_{e}') 
        return parent_info, child_info

def click_next(driver):
    class_name = "slider__control--3gDsk slider__right--1wWef"
    try:
        driver.find_element(By.CLASS_NAME, class_name.replace(' ','.')).click()
        return True
    except: return False
    
def close_note(driver):
    class_name = "anchor__anchor--2QZvA baseModal__close--3j36q" 
    try:
        driver.find_element(By.CLASS_NAME, class_name.replace(' ','.')).click()
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
def find_slider_button(driver):
    class_name2 = "slider__control--3gDsk slider__right--1wWef"
    try:
        next_btt = driver.find_element(By.CLASS_NAME, class_name2.replace(' ','.'))
        return next_btt
    except Exception as e:
        return None

def find_taste(driver, info):
    class_name = "col mobile-column-6 tablet-column-6 desktop-column-4 slider__grid--tKls1"
    try:
        cards = driver.find_elements(By.CLASS_NAME, class_name.replace(' ','.'))
        next_btt = find_slider_button(driver)
        action = ActionChains(driver)
        for i,card in enumerate(cards):
            card = card.find_element(By.TAG_NAME, 'button')
            action.move_to_element(card).click().perform()
            time.sleep(1.5)
            parent_info, child_info = get_tasting_notes(driver)
            info.update(parent_info)
            p_note = list(parent_info.keys())[0].replace(' ','_')
            info[f'{p_note}_child'] = child_info
            if (i % 2 == 1) & (i != len(cards) -1): 
                if next_btt:
                    action.move_to_element(next_btt).click().perform()
                    time.sleep(2)
        return info
    except Exception as e:
        screenshot(driver, f'taste_{e}') 
        return info
#------------------------------------------------------------------------------------------------

        
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
def write_data(write_file, datas):
    write_file = pd.concat([write_file,  pd.DataFrame(datas)], ignore_index=True)
    return write_file

def get_data(driver, urls, done, df):
    datas = []
    for url in tqdm(urls):
        if url not in done:
            datas.append(get_basic_info(driver, url))
            done.add(url)
            if len(datas) == 100:
                print('Saving')
                df = write_data(df, datas)
                df.to_csv('/home/dhkim/winery/data/basic_info.csv', encoding = 'utf-8-sig',index= False)
                datas.clear()
                with open('/home/dhkim/winery/data/basic_done.pkl','wb') as f: pickle.dump(done,f)
    print('Saving')
    df = write_data(df, datas)
    df.to_csv('/home/dhkim/winery/data/basic_info.csv', encoding = 'utf-8-sig',index= False)
    datas.clear()
    with open('/home/dhkim/winery/data/basic_done.pkl','wb') as f: pickle.dump(done,f)


if __name__ == '__main__':
    my_idx = int(sys.argv[-1])
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
    

    with open('/home/dhkim/winery/data/urls.json', 'r') as f: urls = json.load(f)

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

    urls_for_me = split_list(urls, 6)[my_idx]

    try:
        with open('/home/dhkim/winery/data/basic_done.pkl', 'rb') as f: done  = pickle.load(f)
    except:
        done = set()

    #done = set()
    #df = pd.DataFrame()
    try:
        df = pd.read_csv('/home/dhkim/winery/data/basic_info.csv', encoding = 'utf-8-sig')
        done = done | set(df['url'])
    except:
        df = pd.DataFrame(columns = ['winery','grapes','country','region1','region2','region3', 'wine style','allergens'])


    get_data(driver, urls_for_me, done, df)
    