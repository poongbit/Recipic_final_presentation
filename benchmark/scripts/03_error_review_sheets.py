# step11: 최고 모델(Lite0 v3, 전체 INT8)이 독립 테스트셋에서 틀린 사진을 모아 검토용 시트로 저장
import os, sys, csv, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

E = os.environ.get
D = E('R_DRIVE', '/content/drive/MyDrive/recipic_benchmark')
MODEL = E('R_MODEL', f'{D}/v3b/results_indep_v3/models/recipic_lite0_v3_full_int8.tflite')
TEST = E('R_TEST', '/content/indep_test')
OUT = E('R_OUT', '/content/review')
sys.path.insert(0, os.path.dirname(E('R_BENCH', '/content/recipic_benchmark.py')))
import recipic_benchmark as rb

labels = rb.read_labels(Path(MODEL), None)
runner = rb.Runner(Path(MODEL), 2)
rows = []
for ti, label in enumerate(labels):
    d = Path(TEST) / label
    for p in (sorted(d.iterdir()) if d.is_dir() else []):
        y = runner.predict(rb.load_image(p, runner.size, 'center_crop'))
        order = np.argsort(-y)
        rows.append(dict(file=str(p), true=label, pred=labels[order[0]], conf=float(y[order[0]]),
                         second=labels[order[1]], true_rank=int(np.where(order == ti)[0][0]) + 1))
wrong = [r for r in rows if r['pred'] != r['true']]
print(f'전체 {len(rows)}장, 오답 {len(wrong)}장, 정확도 {100 * (1 - len(wrong) / len(rows)):.1f}%')
print('정답이 2순위였던 오답:', sum(r['true_rank'] == 2 for r in wrong))

shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
for i, r in enumerate(wrong):
    r['id'] = i + 1
with open(f'{OUT}/errors.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'true', 'pred', 'conf', 'second', 'true_rank', 'file'])
    w.writeheader(); w.writerows(wrong)

TILE, CAP, COLS = 250, 44, 6
font = ImageFont.load_default(size=17)
def sheet(items, path, caption):
    if not items:
        return
    n_rows = (len(items) + COLS - 1) // COLS
    im = Image.new('RGB', (COLS * TILE, n_rows * (TILE + CAP)), (30, 30, 30))
    dr = ImageDraw.Draw(im)
    for k, r in enumerate(items):
        x0, y0 = (k % COLS) * TILE, (k // COLS) * (TILE + CAP)
        t = Image.open(r['file']).convert('RGB'); t.thumbnail((TILE - 6, TILE - 6))
        im.paste(t, (x0 + (TILE - t.size[0]) // 2, y0 + (TILE - t.size[1]) // 2))
        for j, line in enumerate(caption(r)):
            dr.text((x0 + 5, y0 + TILE + 2 + j * 20), line, fill=(255, 255, 255) if j == 0 else (255, 200, 80), font=font)
    im.save(path, quality=88)

PER = 24
for s in range(0, len(wrong), PER):
    sheet(wrong[s:s + PER], f'{OUT}/errors_{s // PER + 1:02d}.jpg',
          lambda r: (f"#{r['id']} T:{r['true']}", f"P:{r['pred']} {r['conf']:.2f}"))
ban = [r for r in rows if r['true'] == 'banana' and r['pred'] == 'banana']
sheet(ban, f'{OUT}/banana_correct.jpg', lambda r: (f"T:{r['true']}", f"P:{r['pred']} {r['conf']:.2f}"))
print(sorted(os.listdir(OUT)), {f: os.path.getsize(f'{OUT}/{f}') // 1000 for f in os.listdir(OUT)})
if os.path.isdir(D):
    shutil.rmtree(f'{D}/review', ignore_errors=True)
    shutil.copytree(OUT, f'{D}/review')
print('step11 완료')
