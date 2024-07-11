import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
dotenv_path = r"C:\Users\Administrator\Documents\faucet_bots\.env"
load_dotenv(dotenv_path)

# Get the credentials from the environment variables
EMAIL = os.getenv('freebitco_in_email')
PASSWORD = os.getenv('freebitco_in_password')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to your geckodriver executable
gecko_driver_path = r"C:\Users\Administrator\Documents\Downloads\exe\geckodriver.exe"

# Initialize Firefox options
firefox_options = webdriver.FirefoxOptions()
firefox_options.set_preference("dom.webnotifications.enabled", False)
firefox_options.set_preference("dom.push.enabled", False)
firefox_options.set_preference("privacy.trackingprotection.enabled", True)

# Initialize the WebDriver Service
service = FirefoxService(gecko_driver_path)

def take_screenshot(driver, filename):
    """
    Takes a screenshot of the current browser window.

    Parameters:
    driver (WebDriver): The WebDriver instance controlling the browser.
    filename (str): The name of the file to save the screenshot.
    """
    path = os.path.join(os.getcwd(), filename)
    driver.save_screenshot(path)
    logger.info(f"Screenshot saved to {path}")

def login(driver):
    """
    Logs into FreeBitco.in using the provided WebDriver instance.

    Parameters:
    driver (WebDriver): The WebDriver instance controlling the browser.
    """
    try:
        # Click on the "LOGIN" link
        login_link = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.LINK_TEXT, "LOGIN"))
        )
        login_link.click()
        logger.info("Clicked on the LOGIN link")

        # Wait for the login form to be visible
        WebDriverWait(driver, 30).until(
            EC.visibility_of_element_located((By.ID, "login_form"))
        )
        logger.info("Login form is visible")

        # Wait for the login elements to load
        email_field = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "login_form_btc_address"))
        )
        password_field = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "login_form_password"))
        )
        login_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "login_button"))
        )
        logger.info("Login elements located")

        # Ensure elements are interactable
        driver.execute_script("arguments[0].scrollIntoView();", email_field)
        driver.execute_script("arguments[0].scrollIntoView();", password_field)
        driver.execute_script("arguments[0].scrollIntoView();", login_button)
        logger.info("Elements scrolled into view")

        # Enter login credentials
        email_field.send_keys(EMAIL)
        password_field.send_keys(PASSWORD)
        login_button.click()
        logger.info("Login form submitted")

        # Wait for the login process to complete
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.LINK_TEXT, "FREE BTC"))
        )
        logger.info("Login process complete")

    except Exception as e:
        logger.error(f"An error occurred during login: {e}", exc_info=True)
        take_screenshot(driver, "login_error_screenshot.png")
        raise

def log_earnings(session_start_balance, session_end_balance):
    """
    Logs the BTC earnings for the session.

    Parameters:
    session_start_balance (str): The starting BTC balance for the session.
    session_end_balance (str): The ending BTC balance for the session.
    """
    earnings = float(session_end_balance) - float(session_start_balance)
    log_entry = f"Session Earnings: {earnings} BTC (Start: {session_start_balance}, End: {session_end_balance})\n"
    with open("btc_earnings_log.txt", "a") as log_file:
        log_file.write(log_entry)
    logger.info(log_entry)

def wait_until_next_roll(driver):
    """
    Waits until the next roll is available, based on the presence or absence of the countdown timer.

    Parameters:
    driver (WebDriver): The WebDriver instance controlling the browser.
    """
    while True:
        try:
            # Check for the presence of the "ROLL!" button
            roll_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "free_play_form_button"))
            )
            driver.execute_script("arguments[0].scrollIntoView();", roll_button)
            roll_button.click()
            logger.info("Clicked the ROLL! button")

            # Wait for the balance to change
            initial_balance = driver.find_element(By.CSS_SELECTOR, "span#balance").text
            WebDriverWait(driver, 60).until(
                lambda driver: driver.find_element(By.CSS_SELECTOR, "span#balance").text != initial_balance
            )
            new_balance = driver.find_element(By.CSS_SELECTOR, "span#balance").text
            logger.info(f"New balance: {new_balance}")

            # Check if the balance has changed
            if initial_balance != new_balance:
                logger.info("Balance has changed successfully")
                log_earnings(initial_balance, new_balance)
            else:
                logger.warning("Balance did not change")

            # Handle additional banner after roll
            try:
                additional_banner = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'ui-dialog') and contains(@style, 'display: block')]//button[text()='PLAY NOW']"))
                )
                additional_banner.click()
                logger.info("Closed the additional banner")
            except Exception as e:
                logger.info("Additional banner not found or not clickable")

            # Wait for 1 hour and 5 minutes (3900 seconds) before running again
            time.sleep(3900)
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            take_screenshot(driver, "error_screenshot.png")

# Initialize the WebDriver only once
driver = webdriver.Firefox(service=service, options=firefox_options)

try:
    # Navigate to the FreeBitco.in home page
    driver.get("https://freebitco.in/")
    logger.info("Navigated to FreeBitco.in home page")

    # Try logging in
    try:
        login(driver)
    except Exception as login_error:
        logger.info("Handling potential 'too many tries' message")
        # Check for "too many tries" message and wait
        try:
            too_many_tries_message = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'too many tries')]"))
            )
            wait_time = 42  # Example wait time, adjust as necessary
            logger.info(f"Too many tries, waiting for {wait_time} seconds")
            time.sleep(wait_time)
            logger.info("Retrying login after wait time")
            login(driver)
        except Exception as e:
            logger.error(f"An error occurred while waiting: {e}", exc_info=True)
            take_screenshot(driver, "too_many_tries_screenshot.png")
            raise

    while True:
        try:
            # Handle the cookie consent banner using the provided XPath
            try:
                cookie_consent_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div/a[1]"))
                )
                cookie_consent_button.click()
                logger.info("Clicked the cookie consent button")
            except Exception as e:
                logger.info("Cookie consent button not found or not clickable")
                # Try to remove the cookie banner directly
                try:
                    driver.execute_script("document.querySelector('div.cc_banner.cc_container.cc_container--open').style.display='none';")
                    logger.info("Cookie consent banner hidden using JavaScript")
                except Exception as js_error:
                    logger.error(f"An error occurred while hiding cookie banner: {js_error}", exc_info=True)

            wait_until_next_roll(driver)

        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            take_screenshot(driver, "error_screenshot.png")

finally:
    driver.quit()
    logger.info("Driver quit")
