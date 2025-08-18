import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Setup Chrome
options = Options()
options.add_argument("--start-maximized")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Open NVIDIA page
driver.get("https://build.nvidia.com/nvidia/generative-virtual-screening-for-drug-discovery")

wait = WebDriverWait(driver, 20)

# --- STEP 1: Handle cookie popup ---
try:
    cookie_btn = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
    cookie_btn.click()
    print("✅ Cookie popup closed")
except:
    print("ℹ️ No cookie popup appeared")

# --- STEP 2: Input amino acid sequence ---
seq_box = wait.until(EC.presence_of_element_located((By.TAG_NAME, "textarea")))
seq_box.clear()
seq_box.send_keys("ATCATCTTTGGTGTTTCCTCTGATGAATATAGATACAGAAGCGTCATCAAAGCATGCCAACTAGAAGAGGTACATCGTCTCTTCTGCAACGAAGAC")  # example sequence
print("✅ Sequence entered")

# --- STEP 3: Click 'Generate Molecules' ---
generate_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Generate Molecules')]")))
generate_btn.click()
print("✅ Generate button clicked")

# --- STEP 4: Wait for results (adjust sleep if needed) ---
time.sleep(15)

driver.quit()

