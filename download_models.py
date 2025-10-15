
def download_ipadapter():
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="h94/IP-Adapter", ignore_patterns="sdxl_models/*", local_dir="./downloads/")

    from my_ipadapter_model import load_ipadapter

    ip_model = load_ipadapter(device="cpu")
