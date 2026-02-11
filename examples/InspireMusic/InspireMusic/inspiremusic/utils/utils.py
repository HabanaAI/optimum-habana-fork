import os
import sys
import subprocess

def download_model(repo_url: str, output_dir: str = None, token: str = None):
    try:
        if token:
            repo_url = repo_url.replace("https://", f"https://USER:{token}@")
        else:
            repo_url = f"https://www.modelscope.cn/models/iic/{repo_url}"

        cmd = ["git", "clone", repo_url]
        if output_dir:
            cmd.append(output_dir)

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("Success:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)

def align_trans_scp_file(trans, scp):
    trans_dict = {}
    with open(trans, 'r') as f:
        for line in f:
            sec = line.strip().split("\t")
            trans_dict[sec[0]] = sec[1]
    scp_dict = {}
    with open(scp, 'r') as f:
        for line in f:
            sec = line.strip().split(" ")
            scp_dict[sec[0]] = sec[1]
    with open("text", "w") as f:
        for k, v in scp_dict.items():
            f.write("%s\t%s\n"%(k,trans_dict[k]))