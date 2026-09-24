"""下载 convaiinnovations/laya 最新快照(multilingual 全部文件)"""
from huggingface_hub import snapshot_download

p = snapshot_download("convaiinnovations/laya", allow_patterns=["multilingual/*"])
print("SNAPSHOT:", p)
