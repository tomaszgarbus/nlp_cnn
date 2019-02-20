import sys

import matplotlib.pyplot as plt


class ChartGenerator:

    def __init__(self, print_logged=False):
        self.logged = {'iter': []}
        self.print_logged = print_logged

    def log_values(self, iter_overall, new_vals):
        self.logged['iter'].append(iter_overall)
        for (k, v) in new_vals.items():
            w = self.logged.get(k, [])
            w.append(v)
            self.logged[k] = w
        if self.print_logged:
            message = [k + ": " + str(v) for (k, v) in new_vals.items()]
            print(" ".join(message))
            sys.stdout.flush()

    def _build_chart(self, title='', ylim_min=None, ylim_max=None):
        plt.style.use('seaborn-darkgrid')
        plt.figure()
        for (i, column) in enumerate(self.logged.keys()):
            if column != "iter":
                plt.plot('iter', column, data=self.logged, marker='', linewidth=2, alpha=0.7, label=column)

        plt.legend()

        plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("iterations")
        plt.ylabel("value")
        if ylim_min is not None and ylim_max is not None:
            plt.ylim(ylim_min, ylim_max)

    def show_chart(self, title='', ylim_min=None, ylim_max=None):
        self._build_chart(title, ylim_min, ylim_max)
        plt.show()
        plt.close()

    def save_chart(self, fname: str, title='', ylim_min=None, ylim_max=None):
        self._build_chart(title, ylim_min, ylim_max)
        plt.savefig(fname)
        plt.close()
