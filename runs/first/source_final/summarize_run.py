"""Render measured results of runs/first; no training or inferred metrics."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent/".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path(__file__).resolve().parent/"runs/first"
    metrics = json.loads((out/"metrics.json").read_text())
    summary = json.loads((out/"training_summary.json").read_text())
    benchmark = json.loads((out/"prediction_benchmark.json").read_text())
    cli = json.loads((out/"predict_cli_timing.json").read_text(encoding="utf-8-sig"))
    fit = json.loads((out/"training_fit_diagnostic.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={"width_ratios": [1.5, 1]}, constrained_layout=True)
    for ax, head in zip(axes, ("brand", "colour")):
        with (out/f"{head}_confusion.csv").open(encoding="utf8", newline="") as f:
            rows = list(csv.reader(f))
        names = rows[0][1:]
        matrix = np.array([list(map(int, r[1:])) for r in rows[1:]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=9)
        ax.set_yticks(range(len(names)), [f"{n} (n={matrix[i].sum()})" for i, n in enumerate(names)], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{head.title()} - validation counts")
        for i in range(len(names)):
            for j in range(len(names)):
                if matrix[i, j]:
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "black")
        for i, count in enumerate(matrix.sum(1)):
            if count == 0:
                ax.axhspan(i-.5, i+.5, color="grey", alpha=.13)
    figure.suptitle("First run: 10 validation crops / 7 vehicle groups. Shaded rows: no validation support.")
    figure.savefig(out/"confusion_matrices.png", dpi=160)
    plt.close(figure)
    with (out/"history.csv").open(encoding="utf8", newline="") as f:
        history = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for key, label in (("train_loss", "Train"), ("val_loss", "Validation")):
        axes[0].plot(epochs, [float(r[key]) for r in history], label=label)
    axes[0].set(title="Weighted cross-entropy", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    for key, label in (("brand_val_accuracy", "Brand"), ("colour_val_accuracy", "Colour")):
        axes[1].plot(epochs, [float(r[key]) for r in history], label=label)
    axes[1].axvline(summary["best_epoch"], color="grey", linestyle="--", label="Selected checkpoint")
    axes[1].set(title="Validation on 10 crops", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    figure.savefig(out/"learning_curves.png", dpi=160)
    plt.close(figure)
    val_case = next(c for c in benchmark["cases"] if c["annotation_subset"] == "validation")
    report = f'''# CSAI-3210: первое обучение — 26 октября 2026

Обучение завершено на NVIDIA RTX 4070 SUPER, CUDA 12.8 / PyTorch 2.9.1.
Веса: `runs/first/best.pt`, **{summary['weights_bytes']:,} байт**
({summary['weights_bytes']/1e6:.2f} MB; {summary['weights_bytes']/2**20:.2f} MiB).
Время функции обучения вместе с подготовкой и сохранением: **{summary['training_wall_seconds']:.2f} с**;
непосредственно цикл эпох: {summary['epoch_loop_seconds']:.2f} с. Импорт Python-библиотек
до входа в функцию и последующая оценка в это время не входят.

## Что обучалось

Один заранее заданный опыт: 30 эпох, batch=8, AdamW, lr backbone=0.0001,
lr голов=0.0005, weight decay=0.01, cosine scheduler, AMP, seed=42.
Общий RegNet-Y-800MF и два собственных выхода: 15 марок и восемь цветов.
Использованы только ImageNet-веса backbone и отдельный COCO-детектор YOLO11n.
Готовых классификаторов марок и внешних сервисов распознавания не использовано.
Новых запусков для подбора удачного seed или результата не проводилось.

Проверены все 52 активных кропа, размеры и рамки, декодирование и тензоры.
Исходные 76 фотографий сохранены: SHA-256 совпадают с исходным манифестом.
Метки и кропы в этом запуске не изменялись.

## Разбиение

42 кропа / 30 групп машин в train; 10 кропов / 7 групп машин в validation.
Повторные ракурсы одного vehicle_id и все объекты одной фотографии объединены
в связные split-группы; между выборками нет пересечений vehicle_id, photo,
source. Проверено на 100 seed; фактический run использовал только seed=42.
Улучшен выбор holdout: наряду с марками учитывается покрытие цветов, поэтому
в проверке есть white, black и grey. Brown удерживается в train: обе машины
этого цвета относятся к маркам с единственным автомобилем.

Split, пути кропов, копия меток и отпечаток содержимого датасета сохранены
внутри checkpoint и рядом с ним. Evaluation использует именно этот split
и отказывается считать результат при изменении меток или содержимого кропов.
Никакие неизвестные марки не исключаются молча из оценки.

## Измеренные validation-результаты

Лучшая эпоха — **{summary['best_epoch']}**. Выбор checkpoint:
0.6 × brand accuracy + 0.4 × colour accuracy; при равенстве — меньшая val loss.

| Оценка | Марка | Цвет |
|---|---:|---:|
| По 10 кропам | {metrics['brand']['correct']}/10 = {metrics['brand_accuracy']:.1%} | {metrics['colour']['correct']}/10 = {metrics['colour_accuracy']:.1%} |
| По 7 группам машин, средние вероятности ракурсов | {metrics['vehicle_probability_mean']['brand']['correct']}/7 = {metrics['vehicle_probability_mean']['brand']['accuracy']:.1%} | {metrics['vehicle_probability_mean']['colour']['correct']}/7 = {metrics['vehicle_probability_mean']['colour']['accuracy']:.1%} |
| Macro-F1 по представленным в validation классам | {metrics['brand']['macro_f1_observed_classes']:.4f} | {metrics['colour']['macro_f1_observed_classes']:.4f} |

Обе метки одновременно верны на {round(metrics['joint_accuracy']*10)}/10 кропов.
На обучающих 42 кропах тот же checkpoint даёт
{fit['brand']['correct']}/42 по марке и {fit['colour']['correct']}/42 по цвету.
Это проверка запоминания train, а не качества на новых машинах. Разрыв с
validation показывает сильное переобучение. Первый опыт технически работает,
но классификатор марок пока непригоден для надёжного практического распознавания.

Матрицы ошибок: `brand_confusion.csv`, `colour_confusion.csv` и
`confusion_matrices.png`. Все 15/8 классов сохранены в матрицах; пустые
строки означают отсутствие проверки, а не идеальную точность.
Пообъектные ответы и вероятности: `validation_predictions.json`.

### Марки, представленные в validation

| Марка | Кропов | Верно | Recall |
|---|---:|---:|---:|
'''
    for name, values in metrics["brand"]["per_class"].items():
        if values["support"]:
            report += f"| {name} | {values['support']} | {values['correct']} | {values['recall']:.1%} |\n"
    report += f'''
Без независимой проверки марки: **{', '.join(metrics['brands_with_only_one_vehicle'])}**.
У каждой лишь одна группа автомобиля; её ракурсы оставлены в train.
Красный, синий, зелёный и жёлтый отсутствуют во всём датасете. Бежевый/
коричневый представлен двумя train-машинами, но отсутствует в validation.
Для них нельзя заявлять измеренную validation-точность.

Validation крошечная, включает всего семь групп машин и использована для
выбора лучшей эпохи. Это не независимый test set и не оценка всех 15 марок.
Четыре BYD-кропа представляют одну машину, поэтому отдельно показана оценка
по группам. Метки являются визуальной разметкой ассистентом, без независимого
экспертного подтверждения. Нужны дополнительные машины, места и освещение,
особенно для девяти одиночных марок и недостающих цветов.

## Проверка нескольких машин

Фотографии выбраны заранее: photo_27 (train-демонстрация) и photo_76 (validation).
На каждой детектор нашёл две машины, соответствующие двум размеченным объектам
при IoU >= 0.5. Это локальный функциональный тест; неполная разметка фона не
позволяет заявлять detection mAP.

| Фото | Истинные метки | Выход программы | Итог |
|---|---|---|---|
| 27, train | Kia grey; Kia white | Kia grey; Kia white | Обе машины верны, но фото из train |
| 76, validation | Leapmotor black | unknown white; raw top-1 Aion | Марка и цвет неверны |
| 76, validation | Chevrolet white | Kia white | Марка неверна, цвет верен |

Порог марки 0.4 задан заранее. `unknown` означает низкую максимальную
вероятность, а не отдельно обученное обнаружение неизвестной марки.
В метриках классификации используется top-1 без этого порога.
На фото 76 вероятность raw Aion — 0.239, а Kia для Chevrolet — 0.593;
порог не подбирался по этим ошибкам.

Время на photo_76:

- Новый CLI-процесс, включая Python/import, загрузку обеих моделей,
  чтение, инференс и сохранение: **{cli['process_wall_seconds']:.3f} с**.
- Внутри первого CLI: собственно detector + classifier —
  {json.loads((out/'predict_photo_76_cli.json').read_text())['inference_seconds']:.3f} с.
- С загруженными моделями после двух прогревов: медиана
  **{val_case['warm_inference_median_seconds']*1000:.2f} мс** по 10 повторам,
  диапазон {val_case['warm_inference_min_seconds']*1000:.2f}–{val_case['warm_inference_max_seconds']*1000:.2f} мс.
  Включены YOLO, подготовка кропов, классификация двух машин и CUDA synchronize;
  исключены disk I/O, загрузка моделей и старт Python.

Аннотированные производные: `predict_photo_27.jpg`, `predict_photo_76.jpg`.
Подробности и все замеры — `prediction_benchmark.json`,
`predict_photo_76_cli.json`, `predict_cli_timing.json`.

## Исправления и воспроизведение

Исправлены fingerprint/frozen split для оценки, запрет пустой выборки и
молчаливого исключения неизвестных марок, покрытие цветов при выборе holdout,
взвешивание среднего loss по размеру batch и сохранение времени обучения.
Предсказание теперь классифицирует кропы до рисования и отдельной копией
рисует результат, поддерживает Unicode/EXIF и запрещает запись поверх
исходных фото. Несколько кропов классифицируются одним batch на CUDA.

После обучения обнаружена ошибка сериализации метаданных версии PyTorch:
TorchVersion исправлен на обычную строку; значения всех обученных тензоров
совпадают до/после исправления, проверены torch.equal. Повторное обучение не
потребовалось. Checkpoint загружается с weights_only=True без разрешения
произвольной десериализации. Последующие train сохраняют строку сразу.

Восемь регрессионных тестов прошли, включая группировку, 100 split seed,
контроль изменённых данных, отсутствующие классы, сохранение аннотаций
с кириллическими путями и защиту оригиналов.

```powershell
.\\run-project.ps1 evaluate --data data --weights runs/first/best.pt --out runs/first --device cuda
.\\run-project.ps1 predict --image cars_photos/photo_76_2026-10-26_17-10-50.jpg --weights runs/first/best.pt --detector yolo11n.pt --device cuda --save runs/first/new_prediction.jpg --json runs/first/new_prediction.json
```

Первый опыт: `project.py train --data data --epochs 30 --batch 8 --lr 0.0001
--head-lr 0.0005 --seed 42 --device cuda --out runs/first`.
Существующий run защищён от перезаписи; для нового опыта нужен другой --out.

SHA-256 весов: `{metrics['weights_sha256']}`.
Отпечаток датасета: `{metrics['dataset_fingerprint']}`.
'''
    (out/"REPORT.md").write_text(report, encoding="utf8")
    print(out/"REPORT.md")


if __name__ == "__main__":
    main()
