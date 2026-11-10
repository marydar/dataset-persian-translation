
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# API CONFIGURATION
# ============================================================



# Nvidia
# NVIDIA_API_KEY = os.getenv(
#     "NVIDIA_API_KEY"
# )

# BASE_URL = "https://integrate.api.nvidia.com/v1"
# # MODEL = "z-ai/glm-5.3-flash"
# MODEL = "deepseek-ai/deepseek-v4.1-flash"
# MODEL = "deepseek-ai/deepseek-v4-pro-0813"
# MODEL = "moonshotai/kimi-k3"


HOOSHYAR_API_KEY = os.getenv("HOOSHYAR_API_KEY")
BASE_URL = "https://hooshyar.payampardaz.com/api/v1"
# MODEL = "gpt-6-luna"
MODEL = "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL"

REASONING = False

# ============================================================
# GENERATION SETTINGS
# ============================================================

BATCH_SIZE = 1

MAX_TOKENS = 8192

TEMPERATURE = 0.2

# ============================================================
# REQUEST SETTINGS
# ============================================================

LLM_RETRIES = 10

RETRY_SLEEP = 10

REQUEST_DELAY_MIN = 4
REQUEST_DELAY_MAX = 8

REQUESTS_BEFORE_SLEEP = 30
SLEEP_SECONDS = 30

# ============================================================
# SOURCE DATASET
# ============================================================

SOURCE_DATASET = "maryamdar/topic_classification_dataset_gen"

CONFIG_NAME = None

SPLIT = "train"

# Text column to translate
TEXT_COLUMN = "text"

# Label column used for stratified sampling
LABEL_COLUMN = "label"


# ============================================================
# OUTPUT
# ============================================================

CHECKPOINT_PATH = "output/translation_checkpoint.json"

FINAL_DATASET_PATH = "output/translated_dataset"

# New Hugging Face repository
HF_REPO = "maryamdar/topic_classification_dataset_gen_translated"

# ============================================================
# TRANSLATION
# ============================================================

SOURCE_LANGUAGE = "English"

TARGET_LANGUAGE = "Persian"

# ============================================================
# CHECKPOINT / EXPORT SETTINGS
# ============================================================

# Include the original source columns in the output.
PRESERVE_ALL_COLUMNS = True

# Delay between successful requests
USE_RANDOM_DELAY = True


# Dataset filtering
FILTER_USE_FOR_TRAIN = True

EXCLUDED_GENERATOR_MODELS = [
    "MiMo V2.5 Free",
]

# ============================================================
# SAMPLE SELECTION
# ============================================================

START_IDX = 20000

END_IDX = 21000

# Fixed seed for reproducible stratified ordering
SHUFFLE_SEED = 42

# Local cache for the complete stratified dataset
DATASET_CACHE_DIR = "./dataset_cache"