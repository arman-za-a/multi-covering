# نصب و اجرا

## نصب پکیج‌ها

```bash
pip install numpy scipy scikit-learn pandas matplotlib
```

## آماده‌سازی

```bash
mkdir -p results plots/smallset plots/FixedM_large plots/FixedN
echo "instance,paper average optimality gap,average optimality gap,average improvement,average time gap" > results/summary.csv
```

پوشه‌های `instances/` و `paper_results/` باید کنار فایل‌های پایتون باشند.

## اجرا

```bash
python compare.py
```
