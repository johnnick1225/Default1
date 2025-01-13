import os
import time
import pandas as pd
from tqdm import tqdm  # 添加进度条支持
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def initialize_browser():
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # 启用 Headless 模式
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # 设置Chromium浏览器路径
    chromium_path = r'H:\VS Code\chrome-win64\chrome.exe'
    if not os.path.exists(chromium_path):
        raise FileNotFoundError(f"Chromium浏览器不存在于: {chromium_path}")
    chrome_options.binary_location = chromium_path
    
    # 定义可能的ChromeDriver路径
    possible_paths = [
        os.path.join(os.path.dirname(chromium_path), 'chromedriver.exe'),  # Chromium目录
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chromedriver.exe'),  # 当前目录
        r'H:\VS Code\chrome-win64\chromedriver.exe',  # 指定Chromium目录
        r'H:\VS Code\chromedriver-win64\chromedriver.exe',  # 添加新的路径
    ]
    
    # 查找可用的ChromeDriver
    driver_path = None
    for path in possible_paths:
        if os.path.exists(path):
            driver_path = path
            break
    
    if not driver_path:
        raise FileNotFoundError(
            "找不到ChromeDriver，请按照以下步骤操作：\n"
            "1. 确认Chromium版本：查看chrome.exe版本\n"
            "2. 下载对应版本ChromeDriver：https://chromedriver.chromium.org/downloads\n"
            "3. 解压并将chromedriver.exe放在以下任一位置：\n" +
            "\n".join(possible_paths)
        )
    
    print(f"使用Chromium路径: {chromium_path}")
    print(f"使用ChromeDriver路径: {driver_path}")
    
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(60)  # 增加页面加载超时时间
    return driver

def get_literature_info(driver, search_query):
    base_url = "https://webofscience.clarivate.cn/wos/alldb/basic-search"
    try:
        driver.get(base_url)
        wait = WebDriverWait(driver, 20)  # 增加等待时间
        
        # 等待搜索框出现并处理可能的弹窗
        try:
            popup = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'popup-close-button')))
            popup.click()
        except:
            pass
        
        # 等待并输入检索信息
        search_box = wait.until(EC.presence_of_element_located((By.ID, 'searchInputBox')))
        search_box.clear()
        search_box.send_keys(search_query)
        search_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'searchButton')))
        search_button.click()
        
        # 使用显式等待获取文献信息
        title = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.title-value'))).text
        authors = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.authors-value'))).text
        journal = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.source-title-value'))).text
        year = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.publish-year-value'))).text
        
        # 获取PDF链接
        try:
            pdf_link = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'a[data-type="PDF"]')))
            pdf_url = pdf_link.get_attribute('href')
        except:
            pdf_url = None
            
        return {
            'title': title,
            'authors': authors,
            'journal': journal,
            'year': year,
            'pdf_url': pdf_url
        }
        
    except TimeoutException:
        print(f"页面加载超时: {search_query}")
        return None
    except Exception as e:
        print(f"获取文献信息失败: {search_query}, 原因: {str(e)}")
        return None

def download_literature(pdf_url, target_folder, literature_info):
    if not pdf_url:
        return False
        
    try:
        target_folder = os.path.abspath(target_folder)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            
        # 处理文件名中的非法字符
        filename = "".join([c for c in f"{literature_info['title']}_{literature_info['year']}.pdf" 
                       if c.isalnum() or c in (' ', '-', '_', '.')])
        file_path = os.path.join(target_folder, filename)
        
        # 配置下载选项
        options = Options()
        options.add_experimental_option('prefs', {
            'download.default_directory': target_folder,
            'download.prompt_for_download': False,
            'plugins.always_open_pdf_externally': True
        })
        
        download_driver = webdriver.Chrome(options=options)
        try:
            download_driver.get(pdf_url)
            time.sleep(10)  # 增加等待时间
        finally:
            download_driver.quit()
        
        print(f"成功下载: {file_path}")
        return True
        
    except Exception as e:
        print(f"下载失败: {pdf_url}, 原因: {str(e)}")
        return False

def main():
    # 检查必要文件是否存在
    literature_list_path = r'H:\VS Code\Default path\批量下载文献\savedrecs.xls'
    download_folder = r'H:\VS Code\Default path\批量下载文献\downloads'
    log_file_path = r'H:\VS Code\Default path\批量下载文献\无法下载的文献.log'
    
    if not os.path.exists(literature_list_path):
        raise FileNotFoundError(f"文献列表文件不存在: {literature_list_path}")
    
    # 读取文献列表
    df = pd.read_excel(literature_list_path)
    print(f"共找到 {len(df)} 篇文献")
    
    # 初始化浏览器
    driver = initialize_browser()
    
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        for index, row in df.iterrows():
            search_query = row['Title']  # 假设文献列表中有 'Title' 列
            if pd.isna(search_query) or search_query.strip() == "":
                print(f"第 {index+1} 篇文献标题为空，跳过")
                continue
                
            print(f"正在处理第 {index+1} 篇文献: {search_query}")
            literature_info = get_literature_info(driver, search_query)
            
            if literature_info and literature_info['pdf_url']:
                if download_literature(literature_info['pdf_url'], download_folder, literature_info):
                    log_file.write(f"{search_query}\t{literature_info['title']}\t成功\n")
                    continue
                    
            # 记录无法下载的文献
                log_file.write(f"{search_query}\t{literature_info['title'] if literature_info else '获取信息失败'}\t失败\n")
            
    driver.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")