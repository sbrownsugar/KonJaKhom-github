import importlib

mods = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn",
        "streamlit", "joblib", "pythainlp", "requests"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print("  OK   %-14s %s" % (m, v))
    except Exception as e:
        print("  FAIL %-14s %s" % (m, e))

from pythainlp.tokenize import word_tokenize
print("  Thai tokenize:", word_tokenize("ยาละลายลิ่มเลือดกินกับยาแก้ปวดข้อ"))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
print("  sklearn TF-IDF + LogReg: ready")
