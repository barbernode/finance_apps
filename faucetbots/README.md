# FreeBitco.in Automation Script

This script automates the process of logging into FreeBitco.in, rolling for free bitcoins, and logging the earnings. It is designed to run every hour and five minutes, using Selenium to control the browser.

## Features

- Logs into FreeBitco.in using credentials stored in a `.env` file.
- Waits for the free roll countdown to reset.
- Handles cookie consent banners.
- Clicks the "ROLL!" button and logs the earnings.
- Keeps the browser session alive to avoid reopening every hour.

## Dependencies

Make sure you have the following dependencies installed:

- [Python 3.10+](https://www.python.org/downloads/)
- [Selenium](https://pypi.org/project/selenium/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [Geckodriver](https://github.com/mozilla/geckodriver/releases)

You can install the required Python packages using:

```bash
pip install selenium python-dotenv
```

## Premium Membership Requirement

To run this script, you need a premium membership on FreeBitco.in to disable CAPTCHA. Without a premium membership, the script will not be able to bypass the CAPTCHA, and automation will fail.

## Getting Started

1. **Clone the repository**:

```bash
git clone https://github.com/luisvinatea/finance_apps/faucetbots.git
```

2. **Download and install Geckodriver**:

- Download Geckodriver from [here](https://github.com/mozilla/geckodriver/releases).
- Extract the geckodriver executable and place it in a directory.
- Update the `gecko_driver_path` in the script with the path to your Geckodriver executable.

3. **Create a `.env` file**:

Create a `.env` file in the same directory as the script with the following contents:

```
freebitco_in_email=your_email@example.com
freebitco_in_password=your_password
```

4. **Run the script**:

```bash
python C:\Users\Administrator\Documents\faucet_bots\selenium_freebitco_in.py
```

## Usage

- The script will navigate to FreeBitco.in, log in using the provided credentials, and wait for the free roll countdown to reset.
- It will then click the "ROLL!" button and log the earnings.
- The browser session remains open, and the script will rerun every hour and five minutes.

## Referral Link

If you don't have an account yet, you can sign up using my referral link for extra benefits!

[Win Free Bitcoins every hour!](https://freebitco.in/?r=636783)

## Contributing

Feel free to fork this repository and make improvements. Pull requests are welcome!
