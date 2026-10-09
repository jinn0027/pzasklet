# settings.py

# ベースとなるモデルディレクトリのパス
BASE_MODEL_DIR = "/opt/models"

# 利用可能なモデルの定義（キー名: ディレクトリ名）
AVAILABLE_MODELS = {
    # Qwenシリーズ
    "qwen-4b-instruct": f"{BASE_MODEL_DIR}/Qwen/Qwen3-4B-Instruct-2507",
    "qwen-4b-thinking": f"{BASE_MODEL_DIR}/Qwen/Qwen3-4B-Thinking-2507",
    "qwen-3.5-4b": f"{BASE_MODEL_DIR}/Qwen/Qwen3.5-4B",
    "qwen-3.5-9b": f"{BASE_MODEL_DIR}/Qwen/Qwen3.5-9B",
    
    # LLM-jpシリーズ
    "llm-jp-13b-4": f"{BASE_MODEL_DIR}/llm-jp/llm-jp-3.1-13b-instruct4",
    "llm-jp-8b-base": f"{BASE_MODEL_DIR}/llm-jp/llm-jp-4-8b-base",
}

# デフォルトで使用するモデルキー
DEFAULT_MODEL_KEY = "qwen-4b-instruct"

# LLMの共通パラメータ
TEMPERATURE = 0.0
SEED = 42
NUM_TEMPLATE_MAPPING_SELECTIONS = 3
NUM_NORMALIZATION_SELECTIONS = 1
