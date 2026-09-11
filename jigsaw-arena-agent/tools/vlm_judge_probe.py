import base64, io, os
from openai import OpenAI
from PIL import Image

KEY = os.environ["DEEPSEEK_API_KEY"]   # 凭据不入库，从环境变量读取
c = OpenAI(api_key=KEY, base_url="https://api.deepseek.com/v1", timeout=300)
OUT = os.environ.get("CROP_DIR", "crops")
SYS = "你是视觉质检员。看图后直接给结论，理由不超过 80 字，不要长篇推理。"

def enc(path, target=1400, quality=92):
    im = Image.open(path).convert("RGB")
    sc = target / max(im.size)
    im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, format="JPEG", quality=quality)
    url = "da" + "ta:image/" + "jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}, im.size

def ask(title, parts, max_tokens=6000):
    print("=" * 12, title, "=" * 12, flush=True)
    for attempt in range(2):
        try:
            r = c.chat.completions.create(model="deepseek-flash", messages=[{"role": "system", "content": SYS}, {"role": "user", "content": parts}], max_tokens=max_tokens)
            m = r.choices[0].message; txt = (m.content or "").strip()
            print("attempt", attempt, "| finish:", r.choices[0].finish_reason, "| tokens:", r.usage.completion_tokens, flush=True)
            if txt:
                print("CONTENT:", txt[:1200], flush=True); return
            print("(empty content, retrying)", flush=True)
        except Exception as e:
            print("ERR:", str(e)[:300], flush=True)
    print("NO ANSWER", flush=True)

w = os.path.join(OUT, "crop_wrong.png"); r_ = os.path.join(OUT, "crop_right.png")
iw, sz = enc(w, 1400); ir, sz2 = enc(r_, 1400)
print("sent sizes:", sz, sz2, flush=True)
ask("G A/B upscaled", [{"type": "text", "text": "图A、图B是同一块 3x3 拼图板的两次拍摄，一次摆放正确（山羊图案在接缝处连续），一次有拼块放错（接缝处图案错位）。哪一次正确？格式：结论=A或B；理由=..."},
                       {"type": "text", "text": "图A:"}, iw, {"type": "text", "text": "图B:"}, ir])
ask("H wrong-cell", [{"type": "text", "text": "这是 3x3 拼图板照片，其中至少有一块拼块放错位置（该格图案与相邻格不连续）。请指出是哪一格放错了（用左上/上中/右上/左中/中心/右中/左下/下中/右下描述），并说明该格图案应该是什么。"}, enc(w, 1400)])
