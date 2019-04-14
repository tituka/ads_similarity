from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import selenium
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import urllib
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')

#driver = webdriver.Chrome('/usr/bin/chromedriver', chrome_options=options)
search_term="Review BMW X3"

print('https://loop.hypersonica.com/serp?q='+ urllib.parse.quote_plus(search_term))

def proxied(proxy):
    capabilities = DesiredCapabilities.PHANTOMJS.copy()
    capabilities['phantomjs.cli.args'] = ['--proxy=' + proxy, '--proxy-type=http',
        '--proxy-auth=' + os.getenv('WONDERPROXY_USER', "hypersonica") + ':' + os.getenv('WONDERPROXY_PASS',
            "%WToiIGli0S1h0bAMy@x")
        ]

    return webdriver.Remote( desired_capabilities=capabilities)
driver = webdriver.Chrome
driver = proxied('malaysia.wonderproxy.com:11000')
"""Infosonica ad count"""
def infospace_ads(search_term):
    driver.get('https://loop.hypersonica.com/serp?q='+ urllib.parse.quote_plus(search_term))
    driver.save_screenshot("screenshot.png")
    time.sleep(5) # Let the user actually see something!
    try:
        ad_element=driver.find_elements_by_class_name("ads-bing-top__result")
        infosonica_ads=len(ad_element)
    except:
        infosonica_ads=0
    try:
        ad_element=driver.find_elements_by_class_name("ads-bing-bottom__result")
        info_btm=len(ad_element)
    except:
        info_btm=0

    print("bottom ads:" + str(info_btm))

    print(infosonica_ads)
    return(infosonica_ads, info_btm)

#time.sleep(5) # Let the user actually see something!
#search_box = driver.find_element_by_name('q')
"""
search_box.clear()
search_box.send_keys(search_term)
search_box.submit()"""

"""Google custom search ad count"""
def google_custom(search_term):
    driver.get('https://safesearch.hypersonica.com/')

    search_box = driver.find_element_by_name('q')
    search_box.clear()
    search_box.send_keys(search_term)
    search_box.submit()

    print(driver.current_url)
    driver.save_screenshot("screenshot2.png")

    print(driver.page_source)
    els=driver.find_elements_by_xpath("//iframe")
    time.sleep(5) # Let the user actually see something!

    try:
        driver.switch_to_frame("master-1")
       # els2=driver.find_elements_by_xpath("//*[@id=\"adBlock\"]/div[2]")
        left=True
        ads=[]
        google_searches=0
        while left:
            try:
                res=driver.find_element_by_xpath("//*[@id=\"adBlock\"]/div[2]/div["+str(google_searches+1)+"]")
                ads.append(res)
                google_searches+=1
            except:
                left=False
        return google_searches, 0


    except:
        return infospace_ads(search_term)

infospace_top, infospace_btm= infospace_ads(search_term)
g, b=google_custom(search_term)
driver.quit()
print("Infospace searches: "+str(infospace_top))
print("Bottom searches: "+str(infospace_btm))
print("google searches: "+str(g))