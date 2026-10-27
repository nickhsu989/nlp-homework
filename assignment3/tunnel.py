import subprocess
import time
import urllib.request

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def main():
    try:
        subprocess.run(["which", "cloudflared"], check=True, capture_output=True)
    except Exception:
        print("Downloading cloudflared...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, "cloudflared")
        subprocess.run(["chmod", "+x", "cloudflared"], check=True)

    print("Starting tunnel to http://localhost:5000 ...")
    proc = subprocess.Popen(
        ["./cloudflared", "tunnel", "--url", "http://localhost:5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        print(line, end="")
        if "trycloudflare.com" in line:
            print("\nOpen the URL above in your browser.")
            break
    proc.wait()


if __name__ == "__main__":
    main()