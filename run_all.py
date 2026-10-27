# -*- coding: utf-8 -*-
"""Re-render every figure in the repository.

Usage
-----
    python run_all.py
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    'fig1_workflow.py',
    'fig2_radar_violin.py',
    'fig3_oplsda_vip.py',
    'fig4_lasso_models.py',
    'fig5_physchem_mantel.py',
    'fig6_lefse_pca.py',
]


def main():
    sys.path.insert(0, HERE)
    for name in SCRIPTS:
        path = os.path.join(HERE, 'figures', name)
        print(f'--- {name} ---')
        runpy.run_path(path, run_name='__main__')


if __name__ == '__main__':
    main()
