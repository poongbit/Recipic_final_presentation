# step13: 전수 검수 결과(test_audit_v1.csv)로 기존 모델과 새 모델을 같은 정제 테스트셋에서 비교
import os, sys, csv, json
from pathlib import Path
import numpy as np

E = os.environ.get
D = E('R_DRIVE', '/content/drive/MyDrive/recipic_benchmark')
TEST = E('R_TEST', '/content/indep_test')
AUDIT = E('R_AUDIT', f'{D}/review_all/test_audit_v1.csv')
sys.path.insert(0, os.path.dirname(E('R_BENCH', '/content/recipic_benchmark.py')))
import recipic_benchmark as rb

MODELS = {
    '기존 Lite4 (FP32, 46.8MB)': E('R_M_OLD', f'{D}/model_real.tflite'),
    '새 Lite0 v3 (FP32, 13.6MB)': E('R_M_FP32', f'{D}/v3b/recipic_lite0_v3.tflite'),
    '새 Lite0 v3 (INT8, 4.1MB)': E('R_M_INT8', f'{D}/v3b/results_indep_v3/models/recipic_lite0_v3_full_int8.tflite'),
}
audit = list(csv.DictReader(open(AUDIT, encoding='utf-8-sig')))
print('검수 결과:', {v: sum(r['verdict'] == v for r in audit) for v in ('valid_raw', 'valid_cut_or_cooked', 'invalid')})
SUBSETS = {'원본 전체': {'valid_raw', 'valid_cut_or_cooked', 'invalid'},
           '불량 제외': {'valid_raw', 'valid_cut_or_cooked'},
           '생재료만': {'valid_raw'}}
out = {}
for name, path in MODELS.items():
    labels = rb.read_labels(Path(path), None)
    runner = rb.Runner(Path(path), 2)
    for r in audit:
        y = runner.predict(rb.load_image(Path(TEST) / r['label'] / r['file'], runner.size, 'center_crop'))
        r['ok'] = int(labels[int(np.argmax(y))] == r['label'])
    res = {}
    for sname, sel in SUBSETS.items():
        rs = [r for r in audit if r['verdict'] in sel]
        res[sname] = [sum(r['ok'] for r in rs), len(rs)]
    per = {}
    for lab in labels:
        rs = [r for r in audit if r['verdict'] == 'valid_raw' and r['label'] == lab]
        per[lab] = [sum(r['ok'] for r in rs), len(rs)]
    out[name] = {'subsets': res, 'raw_per_class': per}
    print(f'\n{name}')
    for sname, (a, n) in res.items():
        print(f'  {sname}: {a}/{n} = {100 * a / n:.1f}%')
    print('  생재료 클래스별:', {k: f'{a}/{n}' for k, (a, n) in per.items()})
if os.path.isdir(D):
    json.dump(out, open(f'{D}/review_all/audit_results.json', 'w'), ensure_ascii=False, indent=2)
print('step13 완료')
