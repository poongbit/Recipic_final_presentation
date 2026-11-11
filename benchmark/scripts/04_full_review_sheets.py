# step12: 전수 검수용 시트. 최고 모델이 맞힌 사진 전부를 클래스별로 모아 저장 (틀린 94장은 step11에서 검수 완료)
import os, sys, csv, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

E = os.environ.get
D = E('R_DRIVE', '/content/drive/MyDrive/recipic_benchmark')
MODEL = E('R_MODEL', f'{D}/v3b/results_indep_v3/models/recipic_lite0_v3_full_int8.tflite')
TEST = E('R_TEST', '/content/indep_test')
OUT = E('R_OUT', '/content/review_all')
sys.path.insert(0, os.path.dirname(E('R_BENCH', '/content/recipic_benchmark.py')))
import recipic_benchmark as rb

labels = rb.read_labels(Path(MODEL), None)
runner = rb.Runner(Path(MODEL), 2)
correct = []
for ti, label in enumerate(labels):
    d = Path(TEST) / label
    for p in (sorted(d.iterdir()) if d.is_dir() else []):
        y = runner.predict(rb.load_image(p, runner.size, 'center_crop'))
        if int(np.argmax(y)) == ti:
            correct.append(dict(file=str(p), true=label))
for i, r in enumerate(correct):
    r['id'] = i + 1
print('맞힌 사진:', len(correct))
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
with open(f'{OUT}/correct.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'true', 'file']); w.writeheader(); w.writerows(correct)

TILE, CAP, COLS, ROWS, HEAD = 188, 22, 8, 5, 30
font = ImageFont.load_default(size=17); big = ImageFont.load_default(size=22)
n = 0
for label in labels:
    items = [r for r in correct if r['true'] == label]
    for s in range(0, len(items), COLS * ROWS):
        chunk = items[s:s + COLS * ROWS]; n += 1
        rows_n = (len(chunk) + COLS - 1) // COLS
        im = Image.new('RGB', (COLS * TILE, HEAD + rows_n * (TILE + CAP)), (30, 30, 30))
        dr = ImageDraw.Draw(im)
        dr.text((6, 3), f'LABEL: {label}   (#{chunk[0]["id"]} - #{chunk[-1]["id"]})', fill=(255, 220, 90), font=big)
        for k, r in enumerate(chunk):
            x0, y0 = (k % COLS) * TILE, HEAD + (k // COLS) * (TILE + CAP)
            t = Image.open(r['file']).convert('RGB'); t.thumbnail((TILE - 4, TILE - 4))
            im.paste(t, (x0 + (TILE - t.size[0]) // 2, y0 + (TILE - t.size[1]) // 2))
            dr.text((x0 + 4, y0 + TILE), f"#{r['id']}", fill=(255, 255, 255), font=font)
        im.save(f'{OUT}/all_{n:02d}_{label}.jpg', quality=85)
print(n, '장의 시트:', sorted(os.listdir(OUT)))
if os.path.isdir(D):
    shutil.rmtree(f'{D}/review_all', ignore_errors=True)
    shutil.copytree(OUT, f'{D}/review_all')
print('step12 완료')
