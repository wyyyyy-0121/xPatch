"""Regenerate publication figures from bundled, reported aggregate values only."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).resolve().parent
FIG = BASE / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

def save(fig, name):
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight', pad_inches=0.025)
    fig.savefig(FIG / (name + '.png'), dpi=220, bbox_inches='tight', pad_inches=0.025)
    plt.close(fig)

def design():
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    colors = ['#e9f0f4', '#eaf0e5', '#f5eee4']
    entries = [(.69, 'Completed history + origin', 'Fit-only scaling; original targets'),
               (.38, 'Matched implementation arms', 'Default / initialization / anchor / both'),
               (.07, 'Fixed recipe across 20 tasks', 'Persistence, zero, mean and Ridge controls')]
    for i, (y, title, detail) in enumerate(entries):
        ax.add_patch(FancyBboxPatch((.035, y), .93, .235,
                     boxstyle='round,pad=.015', facecolor=colors[i], edgecolor='#526579', linewidth=.8))
        ax.text(.5, y+.157, title, ha='center', va='center', fontsize=8.2, weight='bold')
        ax.text(.5, y+.069, detail, ha='center', va='center', fontsize=7.2)
        if i < 2:
            ax.annotate('', xy=(.5, y-.065), xytext=(.5, y-.017),
                        arrowprops={'arrowstyle': '-|>', 'color': '#526579'})
    save(fig, 'study_design')

def cap():
    with (FIG / 'cap_values.csv').open() as f: rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    x = np.arange(2); width=.22
    for j, (key, label, color) in enumerate([
        ('default20', 'Default, 20 epochs', '#9a8f86'),
        ('default60', 'Default, up to 60', '#688fa6'),
        ('init20', 'Initialized, 20', '#4c806a')]):
        values = [float(r[key]) for r in rows]
        bars = ax.bar(x+(j-1)*width, values, width, label=label, color=color)
        for b, value in zip(bars, values):
            ax.annotate(f'{value:.2f}' if value>2 else f'{value:.4f}',
                        (b.get_x()+b.get_width()/2, value), xytext=(0,3),
                        textcoords='offset points', ha='center', fontsize=6.5)
    ax.axhline(1.8985, color='#303030', linestyle='--', linewidth=.8)
    ax.set_yscale('log'); ax.set_ylim(.9, 30000)
    ax.set_xticks(x, [r['model'] for r in rows]); ax.set_ylabel('Price MSE (USD$^2$, log scale)')
    ax.legend(loc='upper center', fontsize=6.6, frameon=False, ncol=1, bbox_to_anchor=(.53,1.24))
    save(fig, 'cap_comparison')

def annual():
    with (FIG / 'annual_values.csv').open() as f: rows=list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(3.4,2.35))
    x=np.arange(len(rows)); width=.35
    for j,(key,label,color) in enumerate([
        ('repair','Initialized + shrunk','#467f76'),
        ('mean','Shrunk unconditional mean','#99acbf')]):
        values=[float(r[key]) for r in rows]
        bars=ax.bar(x+(j-.5)*width,values,width,label=label,color=color)
        for bar,v in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,v+.7,f'{v:.1f}',ha='center',fontsize=6.7)
    ax.set_xticks(x,[r['label'].replace('|','\n') for r in rows],fontsize=7)
    ax.set_ylabel('MSE improvement (%)');ax.set_ylim(0,62)
    ax.legend(loc='upper left',fontsize=6.8,frameon=False)
    save(fig,'annual_controls')

if __name__ == '__main__':
    FIG.mkdir(exist_ok=True)
    design(); cap(); annual()
