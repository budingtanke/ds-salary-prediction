#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:03:37 2020

@author: Han

Reference: Ömer Sakarya https://github.com/arapfaik/scraping-glassdoor-selenium/blob/master/glassdoor%20scraping.ipynb

"""

from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import ElementNotInteractableException, StaleElementReferenceException


def get_jobs(keyword, num_jobs, verbose, path, sleep_time):
    '''Gathers jobs as a dataframe, scraped from Glassdoor'''

    # Initializing the webdriver
    options = webdriver.ChromeOptions()

    # Uncomment the line below if you'd like to scrape without a new Chrome window every time.
    options.add_argument('headless')

    # Change the path to where chromedriver is in your home folder.
    driver = webdriver.Chrome(
        executable_path=path, options=options)
    driver.set_window_size(1120, 1000)
    url= 'https://www.glassdoor.com/Job/jobs.htm?suggestCount=0&suggestChosen=false&clickSource=searchBtn&typedKeyword=' + keyword + '&locT=N&locId=1&jobType=&context=Jobs&sc.keyword=' + keyword + '&dropdown=0'
    print(url)
    #url = 'https://www.glassdoor.com/Job/jobs.htm?sc.keyword="' + keyword + '"&locT=C&locId=1147401&locKeyword=San%20Francisco,%20CA&jobType=all&fromAge=-1&minSalary=0&includeNoSalaryJobs=true&radius=100&cityId=-1&minRating=0.0&industryId=-1&sgocId=-1&seniorityType=all&companyId=-1&employerSizes=0&applicationType=0&remoteWorkType=0'
    driver.get(url)
    jobs = []

    while len(jobs) < num_jobs:  # If true, should be still looking for new jobs.

        # Let the page load. Change this number based on your internet speed.
        # Or, wait until the webpage is loaded, instead of hardcoding it.
        time.sleep(sleep_time)

        # Test for the "Sign Up" prompt and get rid of it.
        try:
            driver.find_element_by_class_name("selected").click()
            print('selected worked')
        except ElementClickInterceptedException:
            pass

        time.sleep(.1)

        try:
            #driver.find_element_by_class_name("ModalStyle__xBtn___29PT9").click()  # clicking to the X.
            driver.find_element_by_css_selector("[alt='Close']").click()
            print('worked')
        except NoSuchElementException:
            print("failed")
            pass

        # Going through each job in this page
        job_buttons = driver.find_elements_by_class_name(
            "jl")  # jl for Job Listing. These are the buttons we're goi ng to click.
        for job_button in job_buttons:

            print("Progress: {}".format("" + str(len(jobs)) + "/" + str(num_jobs)))
            if len(jobs) >= num_jobs:
                break

            try:
                job_button.click()  # You might
            except ElementNotInteractableException:
                break
            
            time.sleep(1)
            collected_successfully = False

            while not collected_successfully:
                try:
                    company_name = driver.find_element_by_xpath('.//div[@class="employerName"]').text
                    location = driver.find_element_by_xpath('.//div[@class="location"]').text
                    job_title = driver.find_element_by_xpath('.//div[contains(@class, "title")]').text
                    job_description = driver.find_element_by_xpath('.//div[@class="jobDescriptionContent desc"]').text
                    collected_successfully = True
                except:
                    time.sleep(5)

            try:
                salary_estimate = driver.find_element_by_xpath('.//div[@class="salary"]').text
            except (NoSuchElementException, StaleElementReferenceException):
                salary_estimate = -1  # You need to set a "not found value. It's important."

            try:
                rating = driver.find_element_by_xpath('.//span[@class="rating"]').text
            except (NoSuchElementException, StaleElementReferenceException): #https://www.selenium.dev/exceptions/
                rating = -1  # You need to set a "not found value. It's important."

            # Printing for debugging
            if verbose:
                print("Job Title: {}".format(job_title))
                print("Salary Estimate: {}".format(salary_estimate))
                print("Job Description: {}".format(job_description[:500]))
                print("Rating: {}".format(rating))
                print("Company Name: {}".format(company_name))
                print("Location: {}".format(location))

            # Going to the Company tab...
            # clicking on this:
            # <div class="tab" data-tab-type="overview"><span>Company</span></div>
            try:
                driver.find_element_by_xpath('.//div[@class="tab" and @data-tab-type="overview"]').click()

                try:
                    # <div class="infoEntity">
                    #    <label>Headquarters</label>
                    #    <span class="value">San Francisco, CA</span>
                    # </div>
                    headquarters = driver.find_element_by_xpath(
                        './/div[@class="infoEntity"]//label[text()="Headquarters"]//following-sibling::*').text
                except (NoSuchElementException, StaleElementReferenceException):
                    headquarters = -1

                try:
                    size = driver.find_element_by_xpath(
                        './/div[@class="infoEntity"]//label[text()="Size"]//following-sibling::*').text
                except (NoSuchElementException, StaleElementReferenceException):
                    size = -1

                try:
                    founded = driver.find_element_by_xpath(
                        './/div[@class="infoEntity"]//label[text()="Founded"]//following-sibling::*').text
                except (NoSuchElementException, StaleElementReferenceException):
                    founded = -1

                try:
                    type_of_ownership = driver.find_element_by_xpath(
                        './/div[@class="infoEntity"]//label[text()="Type"]//following-sibling::*').text
                except (NoSuchElementException, StaleElementReferenceException):
                    type_of_ownership = -1

                try:
                    industry = driver.find_element_by_xpath(
                        './/div[@class="infoEntity"]//label[text()="Industry"]//following-sibling::*').text
                except (NoSuchElementException, StaleElementReferenceException):
                    industry = -1

                try:
                    sector = driver.find_element_by_xpath(
                        './/div[@class="infoEntity"]//label[text()="Sector"]//following-sibling::*').text
                except (NoSuchElementException, StaleElementReferenceException):
                    sector = -1

                try:
                    revenue = driver.find_element_by_xpath(
                        './/div[@class="infoEntity"]//label[text()="Revenue"]//following-sibling::*').text
                except (NoSuchElementException, StaleElementReferenceException):
                    revenue = -1

                try:
                    competitors = driver.find_element_by_xpath(
                        './/div[@class="infoEntity"]//label[text()="Competitors"]//following-sibling::*').text
                except (NoSuchElementException, StaleElementReferenceException):
                    competitors = -1

            except (NoSuchElementException, StaleElementReferenceException):  # Rarely, some job postings do not have the "Company" tab.
                headquarters = -1
                size = -1
                founded = -1
                type_of_ownership = -1
                industry = -1
                sector = -1
                revenue = -1
                competitors = -1

            if verbose:
                print("Headquarters: {}".format(headquarters))
                print("Size: {}".format(size))
                print("Founded: {}".format(founded))
                print("Type of Ownership: {}".format(type_of_ownership))
                print("Industry: {}".format(industry))
                print("Sector: {}".format(sector))
                print("Revenue: {}".format(revenue))
                print("Competitors: {}".format(competitors))
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            jobs.append({"Job Title": job_title,
                         "Salary Estimate": salary_estimate,
                         "Job Description": job_description,
                         "Rating": rating,
                         "Company Name": company_name,
                         "Location": location,
                         "Headquarters": headquarters,
                         "Size": size,
                         "Founded": founded,
                         "Type of ownership": type_of_ownership,
                         "Industry": industry,
                         "Sector": sector,
                         "Revenue": revenue,
                         "Competitors": competitors})
            # add job to jobs

        # Clicking on the "next page" button
        try:
            driver.find_element_by_xpath('.//li[@class="next"]//a').click()
        except (NoSuchElementException, StaleElementReferenceException):
            print("Scraping terminated before reaching target number of jobs. Needed {}, got {}.".format(num_jobs,
                                                                                                         len(jobs)))
            break

    return pd.DataFrame(jobs)  # This line converts the dictionary object into a pandas DataFrame.

