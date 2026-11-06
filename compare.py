import subprocess
import sys
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

_RUNNER = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
module_name, instance, problem = sys.argv[2], sys.argv[3], sys.argv[4]
module = __import__(module_name)
result = module.solve_gmc(instance, problem)
print(json.dumps(result))
"""

def run(module_name, instance, problem):
    # solve_gmc مقاله و جدید را در یک process جدید اجرا می‌کند
    out = subprocess.check_output(
        [sys.executable, "-c", _RUNNER, PROJECT_DIR, module_name, instance, problem],
        cwd=PROJECT_DIR,
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(out.strip().splitlines()[-1])

def time_gap(paper_time, new_time):
    # اختلاف زمان اجرای هیورستیک جدید و هیورستیک مقاله
    return round(new_time - paper_time, 2)

def area_gap(paper_area, new_area):
    # اختلاف مساحت هیورستیک جدید و هیورستیک مقاله
    return round(new_area - paper_area, 2)

def improvement(paper_area, new_area):
    # بهبود مساحت هیورستیک جدید نسبت به هیورستیک مقاله
    return f"{(paper_area - new_area) / paper_area * 100:.2f} %" if paper_area != 0 else f"{paper_area - new_area}"

def optimality_gap(optimal_area, area):
    # اختلاف مساحت هیورستیک جدید و مساحت بهینه
    return f"{(area - optimal_area) / area * 100:.2f} %" if area != 0 else f"{area - optimal_area}"

def plot_instance_results(
    instance, 
    problems, 
    paper_times, 
    new_times, 
    paper_areas, 
    new_areas, 
    optimal_areas
):
    # سه نمودار time، total area و optimality gap
    def problem_size(problem):
        m, n, _ = re.match(r"problem_(\d+)_(\d+)_(\d+)", problem).groups()
        return int(m), int(n)
    sizes = [problem_size(problem) for problem in problems]
    ms = np.array([size[0] for size in sizes])
    ns = np.array([size[1] for size in sizes])
    if len(set(ns)) == 1 and len(set(ms)) > 1:
        x, xlabel = ms, "number of drones m"
    else:
        x, xlabel = ns, "number of assets n"
    paper_optimality_gaps = [paper_area - optimal_area for paper_area, optimal_area in zip(paper_areas, optimal_areas)]
    new_optimality_gaps = [new_area - optimal_area for new_area, optimal_area in zip(new_areas, optimal_areas)]
    def curve(values):
        values = np.asarray(values, dtype=float)
        xs = np.unique(x)
        mean = np.array([values[x == v].mean() for v in xs])
        low = np.array([values[x == v].min() for v in xs])
        high = np.array([values[x == v].max() for v in xs])
        return xs, mean, low, high
    panels = [
        ("time [s]", [("paper heuristic", paper_times, "tab:orange", "-"),
                      ("new heuristic", new_times, "tab:blue", "-")]),
        ("total area [m$^2$]", [("paper heuristic", paper_areas, "tab:orange", "-"),
                                ("new heuristic", new_areas, "tab:blue", "-"),
                                ("optimal", optimal_areas, "tab:green", "-")]),
        ("optimality gap", [("paper heuristic", paper_optimality_gaps, "tab:orange", "-"),
                            ("new heuristic", new_optimality_gaps, "tab:blue", "-")]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, (ylabel, lines) in zip(axes, panels):
        for label, values, color, style in lines:
            xs, mean, low, high = curve(values)
            ax.plot(xs, mean, style, color=color, label=label)
            ax.fill_between(xs, low, high, color=color, alpha=0.15)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title(f"{instance}: lower is better")
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(f"plots/{instance}.png", dpi=150)

def get_paper_result(instance, problem):
    df = pd.read_csv(f"paper_results/{instance}.csv")
    df = df[((df["greedy_objective_value"] != 0) | (df["objective_value"] == 0)) & (df["name"] == problem)]
    df = df[["greedy_objective_value", "objective_value"]]
    if df.empty:
        return None
    df = df.iloc[0]
    return dict(heuristic_area=float(df["greedy_objective_value"]), optimal_area=float(df["objective_value"]))

def compare_problem_results(instance, problem):
    # مقایسه نتیجه هیوریستیک مقاله، هیوریستیک جدید و جواب بهینه در یک مسئله
    paper_result = get_paper_result(instance, problem)
    if paper_result is None:
        print("there is no paper result for this problem")
        return
    optimal_area = paper_result["optimal_area"]
    paper_heuristic_result = run("paper_heuristic", instance, problem)
    new_heuristic_result = run("new_heuristic", instance, problem)
    paper_feasibility = paper_heuristic_result["feasible"]
    paper_area = paper_result["heuristic_area"]
    paper_time = paper_heuristic_result["time_s"]
    paper_optimality_gap = optimality_gap(optimal_area, paper_area)
    new_feasibility = new_heuristic_result["feasible"]
    new_area = new_heuristic_result["area"]
    new_time = new_heuristic_result["time_s"]
    new_optimality_gap = optimality_gap(optimal_area, new_area)
    print("optimal area:", optimal_area)
    print("paper heuristic:", dict(
        feasible=paper_feasibility, 
        area=paper_area, 
        time_s=paper_time, 
        optimality_gap=paper_optimality_gap
    ))
    print("new heuristic", dict(
        feasible=new_feasibility, 
        area=new_area, 
        time_s=new_time, 
        optimality_gap=new_optimality_gap,
        improvements=new_heuristic_result["improvements"]
    ))
    print("time gap:", time_gap(paper_time, new_time))
    print("improvement:", improvement(paper_area, new_area))

def compare_instance_results(instance):
    # مقایسه میانگین نتایج هیوریستیک مقاله، هیوریستیک جدید و جواب بهینه در مسائل یک نمونه
    paper_df = pd.read_csv(f"paper_results/{instance}.csv")
    paper_df = paper_df[(paper_df["greedy_objective_value"] != 0) | (paper_df["objective_value"] == 0)]
    paper_df = paper_df[["name", "greedy_objective_value", "objective_value"]]
    paper_df = paper_df.drop_duplicates(subset=["name"])
    paper_df[["m", "n", "i"]] = paper_df["name"].str.extract(r"problem_(\d+)_(\d+)_(\d+)").astype(int)
    paper_df = paper_df.sort_values(by=["m", "n", "i"])
    paper_df = paper_df.drop(columns=["m", "n", "i"])
    df = pd.DataFrame(columns=["problem", "feasible", "disks", "area", "time", "optimal area", "area gap", "time gap", "optimality gap", "improvement"])
    problems = []
    optimal_areas = []
    paper_areas = [] 
    paper_times = []
    new_areas = [] 
    new_times = [] 
    for index, row in enumerate(paper_df.itertuples()):
        problem = row.name
        optimal_area = row.objective_value
        paper_heuristic_result = run("paper_heuristic", instance, problem)
        new_heuristic_result = run("new_heuristic", instance, problem)
        paper_feasibility = paper_heuristic_result["feasible"]
        paper_area = row.greedy_objective_value
        paper_time = paper_heuristic_result["time_s"]
        paper_optimality_gap = optimality_gap(optimal_area, paper_area)
        new_feasibility = new_heuristic_result["feasible"]
        new_disks = new_heuristic_result["disks"]
        new_area = new_heuristic_result["area"]
        new_time = new_heuristic_result["time_s"]
        new_optimality_gap = optimality_gap(optimal_area, new_area)
        df.at[index, "problem"] = problem
        df.at[index, "feasible"] = new_feasibility
        df.at[index, "disks"] = new_disks
        df.at[index, "area"] = new_area
        df.at[index, "time"] = new_time
        df.at[index, "optimal area"] = optimal_area
        df.at[index, "area gap"] = area_gap(paper_area, new_area)
        df.at[index, "time gap"] = time_gap(paper_time, new_time)
        df.at[index, "optimality gap"] = optimality_gap(optimal_area, new_area)
        df.at[index, "improvement"] = improvement(paper_area, new_area)
        problems.append(problem)
        optimal_areas.append(optimal_area)
        paper_areas.append(paper_area)
        paper_times.append(paper_time)
        new_areas.append(new_area)
        new_times.append(new_time)
        print("problem:", problem)
        print("optimal area:", optimal_area)
        print("paper heuristic:", dict(
            feasible=paper_feasibility, 
            area=paper_area, 
            time_s=paper_time, 
            optimality_gap=paper_optimality_gap
        ))
        print("new heuristic", dict(
            feasible=new_feasibility, 
            area=new_area, 
            time_s=new_time, 
            optimality_gap=new_optimality_gap,
            improvements=new_heuristic_result["improvements"]
        ))
        print("time gap:", time_gap(paper_time, new_time))
        print("improvement:", improvement(paper_area, new_area))
        print("-" * 125)
    df.to_csv(f"results/{instance}.csv", index=False)
    plot_instance_results(instance, problems, paper_times, new_times, paper_areas, new_areas, optimal_areas)
    n_datasets = len(paper_df)
    paper_avg_area = sum(paper_areas) / n_datasets
    paper_avg_time = sum(paper_times) / n_datasets
    new_avg_area = sum(new_areas) / n_datasets
    new_avg_time = sum(new_times) / n_datasets
    optimal_avg_area = sum(optimal_areas) / n_datasets
    summary_df = pd.read_csv("results/summary.csv")
    summary_df.loc[{"smallset": 0, "FixedM_large": 1, "FixedN": 2}[instance]] = [
        instance, 
        optimality_gap(optimal_avg_area, paper_avg_area), 
        optimality_gap(optimal_avg_area, new_avg_area), 
        improvement(paper_avg_area, new_avg_area),
        time_gap(paper_avg_time, new_avg_time)
    ]
    summary_df.to_csv("results/summary.csv", index=False)
    print("instance:", instance)
    print("paper average optimality gap:", optimality_gap(optimal_avg_area, paper_avg_area))
    print("average optimality gap:", optimality_gap(optimal_avg_area, new_avg_area))
    print("average time gap:", time_gap(paper_avg_time, new_avg_time))
    print("average improvement:", improvement(paper_avg_area, new_avg_area))

if __name__ == "__main__":
    compare_instance_results("smallset")
    compare_instance_results("FixedM_large")
    compare_instance_results("FixedN")