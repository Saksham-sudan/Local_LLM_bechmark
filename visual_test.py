import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

colour_pallet = {'math': '#8b5cf6', 'coding': '#10b981', 'logic': '#334155', 'knowledge': '#f59e0b', 'data_extraction': '#e11d48'}

def bar_plot(in_csv):
    data_in = pd.read_csv(in_csv)
    quality = (data_in.groupby('category')["passed"].mean()) * 100
    bar_plot = sns.barplot(x=quality.index, y=quality, palette= colour_pallet, hue=quality.index, legend=False)
    bar_plot.set(ylabel="passed %")
    labels = [label.get_text() for label in bar_plot.get_xticklabels()]
    labels = ['data extraction' if label == 'data_extraction' else label for label in labels]
    bar_plot.set_xticklabels(labels)
    for cont in bar_plot.containers:
      bar_plot.bar_label(cont)
    plt.ylim(0,100)
    plt.show()

def reg_plot(in_csv):
   data_in = pd.read_csv(in_csv)
   reg_plot = sns.regplot(x= data_in.prompt_length, y= data_in.ttft_ms)
   reg_plot.set(ylabel = "ttft in millisecound", xlabel = "prompt length")
   plt.show()


def visual_func(in_csv):
   bar_plot(in_csv)
   reg_plot(in_csv)

visual_func("benchmark.csv")