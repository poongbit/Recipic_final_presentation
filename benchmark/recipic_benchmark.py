#!/usr/bin/env python3
"""Recipic 분류 모델 경량화 벤치마크.

이미 가지고 있는 FP32 TFLite 분류 모델(예: model_real.tflite) 하나로 다음을 한 번에 수행합니다.

  1. 학습에 쓰지 않은 폴더(기본: 데이터셋의 test)로 정확도를 다시 측정
  2. 모델을 양자화 (가중치 INT8 / 전체 INT8)
  3. 원본과 양자화 모델의 정확도, 파일 크기, CPU 지연 시간을 비교
  4. 결과를 benchmark.md, benchmark.json, 혼동 행렬 CSV로 저장

다시 학습하지 않습니다. GPU도 필요 없습니다.

설치:
    pip install litert-quantizer ai-edge-litert pillow numpy

실행:
    python recipic_benchmark.py --model model_real.tflite --data ./dataset

--data 폴더는 Kaggle "Fruits and Vegetables Image Recognition" 데이터셋의 구조를 따릅니다.
    dataset/
      train/apple/*.jpg ...        (양자화 보정용, 중복 제거 기준)
      test/apple/*.jpg ...         (평가용)
      validation/apple/*.jpg ...   (--eval-split validation 으로 선택 가능)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_PADDING = 32  # MediaPipe Model Maker의 평가 전처리와 같은 값


# --------------------------------------------------------------------------- 라이브러리 로딩
def load_interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "ai-edge-litert"
    except ImportError:
        pass
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        return Interpreter, "tensorflow.lite"
    except ImportError:
        sys.exit("TFLite 인터프리터가 없습니다.  pip install ai-edge-litert")


def load_quantizer():
    try:
        from litert_quantizer import quantizer, recipe
        return quantizer, recipe, "litert-quantizer"
    except ImportError:
        pass
    try:
        from ai_edge_quantizer import quantizer, recipe  # 이전 패키지 이름
        return quantizer, recipe, "ai-edge-quantizer"
    except ImportError:
        sys.exit("양자화 라이브러리가 없습니다.  pip install litert-quantizer")


def pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "?"


# --------------------------------------------------------------------------- 데이터
def read_labels(model_path: Path, labels_file: Path | None) -> list[str]:
    """모델 파일 뒤에 붙어 있는 labels.txt를 읽습니다. 없으면 --labels 파일을 씁니다."""
    if labels_file:
        text = labels_file.read_text(encoding="utf-8")
    elif zipfile.is_zipfile(model_path):
        with zipfile.ZipFile(model_path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not names:
                sys.exit("모델 안에 라벨 파일이 없습니다. --labels 로 지정하세요.")
            text = z.read(names[0]).decode("utf-8")
    else:
        sys.exit("모델에 라벨이 들어 있지 않습니다. --labels 로 라벨 파일(한 줄에 하나)을 지정하세요.")
    labels = [line.strip() for line in text.splitlines() if line.strip()]
    return labels


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT and not p.name.startswith("."))


def find_class_dir(split_dir: Path, label: str) -> Path | None:
    """라벨과 이름이 같은 폴더를 대소문자·공백·밑줄 차이를 무시하고 찾습니다."""
    if not split_dir.is_dir():
        return None
    key = label.lower().replace("_", " ").strip()
    for d in split_dir.iterdir():
        if d.is_dir() and d.name.lower().replace("_", " ").strip() == key:
            return d
    return None


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_image(path: Path, size: int, mode: str) -> np.ndarray:
    """이미지를 [0, 1] 범위의 float32 (size, size, 3) 배열로 만듭니다.

    center_crop: 짧은 변의 size/(size+32) 만큼 가운데를 자른 뒤 리사이즈 (Model Maker 평가 방식)
    squash:      자르지 않고 그대로 리사이즈
    """
    img = Image.open(path).convert("RGB")
    if mode == "center_crop":
        w, h = img.size
        crop = int(size / (size + CROP_PADDING) * min(w, h))
        left, top = (w - crop + 1) // 2, (h - crop + 1) // 2
        img = img.crop((left, top, left + crop, top + crop))
    img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- 추론
class Runner:
    """TFLite 모델 하나를 감싸서 float 입력 → float 점수로 쓰게 해 줍니다."""

    def __init__(self, model_path: Path, threads: int):
        Interpreter, _ = load_interpreter_class()
        self.it = Interpreter(model_path=str(model_path), num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.size = int(self.inp["shape"][1])

    def _to_input(self, x: np.ndarray) -> np.ndarray:
        dtype = self.inp["dtype"]
        if dtype == np.float32:
            return x[None].astype(np.float32)
        scale, zero = self.inp["quantization"]
        info = np.iinfo(dtype)
        q = np.round(x / scale + zero).clip(info.min, info.max)
        return q[None].astype(dtype)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.it.set_tensor(self.inp["index"], self._to_input(x))
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])[0]
        if self.out["dtype"] != np.float32:
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y

    def latency_ms(self, x: np.ndarray, warmup: int, runs: int) -> dict:
        data = self._to_input(x)
        self.it.set_tensor(self.inp["index"], data)
        for _ in range(warmup):
            self.it.invoke()
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            self.it.invoke()
            times.append((time.perf_counter() - t) * 1000)
        times = np.array(times)
        return {"mean": float(times.mean()), "p50": float(np.percentile(times, 50)),
                "p90": float(np.percentile(times, 90))}


# --------------------------------------------------------------------------- 양자화
def quantize(model_path: Path, variant: str, out_path: Path, calib: list[np.ndarray], threads: int) -> None:
    quantizer, recipe, _ = load_quantizer()
    recipes = {
        "weights_int8": recipe.dynamic_wi8_afp32,   # 가중치만 INT8, 연산은 float
        "full_int8": recipe.static_wi8_ai8,         # 가중치와 활성값 모두 INT8 (보정 데이터 필요)
    }
    qt = quantizer.Quantizer(str(model_path))
    qt.load_quantization_recipe(recipes[variant]())
    calib_result = None
    if qt.need_calibration:
        Interpreter, _ = load_interpreter_class()
        sig = Interpreter(model_path=str(model_path)).get_signature_list()
        sig_key = next(iter(sig))
        input_name = sig[sig_key]["inputs"][0]
        data = {sig_key: [{input_name: x[None].astype(np.float32)} for x in calib]}
        calib_result = qt.calibrate(data, num_threads=threads)
    result = qt.quantize(calib_result, enable_progress_report=False)
    result.export_model(str(out_path), overwrite=True)


def attach_labels(model_path: Path, labels: list[str]) -> None:
    """양자화 과정에서 떨어져 나간 labels.txt를 다시 모델 파일 뒤에 붙입니다."""
    if zipfile.is_zipfile(model_path):
        return
    with zipfile.ZipFile(model_path, "a") as z:
        z.writestr("labels.txt", "\n".join(labels) + "\n")


# --------------------------------------------------------------------------- 본체
def main() -> None:
    ap = argparse.ArgumentParser(description="Recipic 분류 모델 평가·양자화 벤치마크")
    ap.add_argument("--model", required=True, type=Path, help="FP32 TFLite 분류 모델")
    ap.add_argument("--data", required=True, type=Path, help="train/ test/ 폴더가 있는 데이터셋 경로")
    ap.add_argument("--out", type=Path, default=Path("benchmark_results"))
    ap.add_argument("--labels", type=Path, default=None, help="모델에 라벨이 없을 때만 지정")
    ap.add_argument("--eval-split", default="test", help="평가에 쓸 하위 폴더 (기본 test)")
    ap.add_argument("--calib-split", default="train", help="양자화 보정에 쓸 하위 폴더 (기본 train)")
    ap.add_argument("--calib-per-class", type=int, default=20)
    ap.add_argument("--preprocess", choices=["center_crop", "squash"], default="center_crop")
    ap.add_argument("--threads", type=int, default=1, help="지연 시간 측정에 쓸 CPU 스레드 수")
    ap.add_argument("--runs", type=int, default=50, help="지연 시간 측정 반복 횟수")
    ap.add_argument("--keep-train-duplicates", action="store_true",
                    help="학습 폴더와 내용이 같은 평가 이미지도 그대로 평가에 포함")
    args = ap.parse_args()

    if not args.model.is_file():
        sys.exit(f"모델 파일이 없습니다: {args.model}")
    if not (args.data / args.eval_split).is_dir():
        sys.exit(f"평가 폴더가 없습니다: {args.data / args.eval_split}")

    out_dir: Path = args.out
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    labels = read_labels(args.model, args.labels)
    fp32 = Runner(args.model, args.threads)
    n_out = int(fp32.out["shape"][-1])
    if len(labels) != n_out:
        sys.exit(f"라벨 수({len(labels)})와 모델 출력 수({n_out})가 다릅니다.")
    size = fp32.size
    print(f"모델: {args.model.name}  입력 {size}x{size}  클래스 {len(labels)}종")
    print(f"라벨: {', '.join(labels)}")

    # ----- 평가 이미지 모으기 -------------------------------------------------
    train_hashes: set[str] = set()
    calib_paths: list[Path] = []
    for label in labels:
        d = find_class_dir(args.data / args.calib_split, label)
        imgs = list_images(d) if d else []
        train_hashes.update(md5(p) for p in imgs)
        if imgs:
            step = max(1, len(imgs) // args.calib_per_class)
            calib_paths += imgs[::step][: args.calib_per_class]

    eval_items: list[tuple[Path, int]] = []
    counts = {}
    removed_dup = 0
    missing = []
    seen: set[str] = set()
    for idx, label in enumerate(labels):
        d = find_class_dir(args.data / args.eval_split, label)
        if d is None:
            missing.append(label)
            counts[label] = 0
            continue
        kept = 0
        for p in list_images(d):
            h = md5(p)
            if h in seen:                       # 평가 폴더 안에서의 중복
                removed_dup += 1
                continue
            if h in train_hashes and not args.keep_train_duplicates:
                removed_dup += 1                # 학습 폴더에 같은 파일이 있음
                continue
            seen.add(h)
            eval_items.append((p, idx))
            kept += 1
        counts[label] = kept
    if missing:
        print(f"[주의] {args.eval_split} 폴더에 없는 클래스: {', '.join(missing)}")
    if not eval_items:
        sys.exit("평가할 이미지가 없습니다. 폴더 이름이 라벨과 같은지 확인하세요.")
    print(f"평가 이미지 {len(eval_items)}장 (중복으로 제외 {removed_dup}장), 보정 이미지 {len(calib_paths)}장")

    # ----- 양자화 -----------------------------------------------------------
    models: dict[str, Path] = {"fp32": args.model}
    notes: dict[str, str] = {}
    calib = [load_image(p, size, args.preprocess) for p in calib_paths]
    for variant in ("weights_int8", "full_int8"):
        if variant == "full_int8" and not calib:
            notes[variant] = f"{args.calib_split} 폴더에 보정 이미지가 없어 건너뜀"
            print(f"[건너뜀] {variant}: {notes[variant]}")
            continue
        out_path = out_dir / "models" / f"{args.model.stem}_{variant}.tflite"
        print(f"양자화: {variant} ...")
        try:
            quantize(args.model, variant, out_path, calib, max(1, os.cpu_count() or 1))
            attach_labels(out_path, labels)
            models[variant] = out_path
        except Exception as e:  # 한 방식이 실패해도 나머지는 계속 측정
            notes[variant] = f"실패: {type(e).__name__}: {e}"
            print(f"[실패] {variant}: {e}")

    # ----- 평가 -------------------------------------------------------------
    print("이미지 불러오는 중 ...")
    images = [load_image(p, size, args.preprocess) for p, _ in eval_items]
    truth = np.array([t for _, t in eval_items])
    results = {}
    preds_by_model = {}
    for name, path in models.items():
        runner = fp32 if name == "fp32" else Runner(path, args.threads)
        preds = np.array([int(np.argmax(runner.predict(x))) for x in images])
        preds_by_model[name] = preds
        conf = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(truth, preds):
            conf[t, p] += 1
        per_class = {}
        for i, label in enumerate(labels):
            n = int(conf[i].sum())
            per_class[label] = {"n": n, "correct": int(conf[i, i]),
                                "accuracy": (conf[i, i] / n) if n else None}
        lat = runner.latency_ms(images[0], warmup=5, runs=args.runs)
        results[name] = {
            "file": str(path),
            "size_mb": path.stat().st_size / 1e6,
            "input_dtype": np.dtype(runner.inp["dtype"]).name,
            "correct": int((preds == truth).sum()),
            "total": int(len(truth)),
            "accuracy": float((preds == truth).mean()),
            "agreement_with_fp32": float((preds == preds_by_model["fp32"]).mean()),
            "latency_ms": lat,
            "per_class": per_class,
        }
        with open(out_dir / f"confusion_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["실제 \\ 예측"] + labels)
            for i, label in enumerate(labels):
                w.writerow([label] + conf[i].tolist())
        print(f"  {name:13s} 정확도 {results[name]['accuracy']*100:5.1f}% "
              f"({results[name]['correct']}/{len(truth)})  "
              f"{results[name]['size_mb']:5.1f}MB  {lat['mean']:6.1f}ms")

    # ----- 저장 -------------------------------------------------------------
    _, interp_pkg = load_interpreter_class()
    _, _, quant_pkg = load_quantizer()
    env = {
        "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "threads": args.threads,
        interp_pkg: pkg_version(interp_pkg),
        quant_pkg: pkg_version(quant_pkg),
    }
    summary = {
        "model": args.model.name, "labels": labels, "input_size": size,
        "preprocess": args.preprocess, "eval_split": args.eval_split,
        "eval_images": len(eval_items), "eval_per_class": counts,
        "duplicates_removed": removed_dup, "missing_classes": missing,
        "calibration_images": len(calib_paths), "notes": notes,
        "environment": env, "results": results,
    }
    (out_dir / "benchmark.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    title = {"fp32": "FP32 (원본)", "weights_int8": "가중치 INT8", "full_int8": "전체 INT8"}
    base = results["fp32"]
    md = [f"# 분류 모델 경량화 벤치마크 — {args.model.name}", "",
          f"- 측정일: {env['date']}",
          f"- 평가 데이터: `{args.eval_split}` 폴더 {len(eval_items)}장, {len(labels) - len(missing)}종 "
          f"(학습 폴더와 같은 파일·중복 {removed_dup}장 제외)",
          f"- 전처리: {args.preprocess}, {size}x{size}, 픽셀값 0~1",
          f"- 양자화 보정: `{args.calib_split}` 폴더 {len(calib_paths)}장",
          f"- 지연 시간: {env['os']} {env['machine']} CPU, 스레드 {args.threads}개, {args.runs}회 평균 "
          f"(스마트폰 실측값이 아님)", "",
          "| 모델 | 크기 | 정확도 | FP32와 예측 일치 | 지연 시간(평균) | p90 |",
          "|---|---|---|---|---|---|"]
    for name, r in results.items():
        size_note = "" if name == "fp32" else f" ({r['size_mb'] / base['size_mb'] * 100:.0f}%)"
        md.append(f"| {title[name]} | {r['size_mb']:.1f}MB{size_note} | "
                  f"{r['accuracy']*100:.1f}% ({r['correct']}/{r['total']}) | "
                  f"{r['agreement_with_fp32']*100:.1f}% | {r['latency_ms']['mean']:.1f}ms | "
                  f"{r['latency_ms']['p90']:.1f}ms |")
    for name, note in notes.items():
        md.append(f"| {title[name]} | {note} | | | | |")
    md += ["", "## 클래스별 정확도", "", "| 클래스 | 장수 | " + " | ".join(title[n] for n in results) + " |",
           "|---|---|" + "---|" * len(results)]
    for label in labels:
        n = results["fp32"]["per_class"][label]["n"]
        cells = []
        for name in results:
            acc = results[name]["per_class"][label]["accuracy"]
            cells.append("-" if acc is None else f"{acc*100:.0f}%")
        md.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    md += ["", "혼동 행렬은 `confusion_*.csv`, 양자화된 모델은 `models/` 폴더에 있습니다."]
    (out_dir / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n결과 저장: {out_dir / 'benchmark.md'}")


if __name__ == "__main__":
    main()
